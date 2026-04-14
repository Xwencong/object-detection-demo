package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/gif"
	_ "image/png"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "golang.org/x/image/tiff"
)

// Config 对应 config.json 的结构，所有字段均可选。
type Config struct {
	Input   string `json:"input"`
	Output  string `json:"output"`
	Classes string `json:"classes"`
	Labels  *bool  `json:"labels"`
	Verbose *bool  `json:"verbose"`
}

func loadConfig(path string) Config {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		fmt.Printf("警告: 解析 %s 失败: %v\n", path, err)
	}
	return cfg
}

// loadEnv 解析 .env 文件，将键值对写入全局变量。
// 优先级低于 config.json 和命令行参数。
func loadEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return // 文件不存在时静默跳过
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.IndexByte(line, '=')
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])
		// 去掉行内注释
		if ci := strings.Index(val, " #"); ci >= 0 {
			val = strings.TrimSpace(val[:ci])
		}
		switch key {
		case "USE_CPU":
			UseCPU = val == "true"
		case "MODEL_PATH":
			if val != "" {
				ModelPath = val
			}
		case "INPUT_IMAGE_PATH":
			if val != "" {
				InputImagePath = val
			}
		case "OUTPUT_IMAGE_PATH":
			if val != "" {
				OutputImagePath = val
			}
		case "BATCH_SIZE":
			if n, err := strconv.Atoi(val); err == nil && n > 0 {
				BatchSize = n
			}
		case "SAVE_WORKERS":
			if n, err := strconv.Atoi(val); err == nil && n > 0 {
				SaveWorkers = n
			}
		case "DRAW_LABELS":
			DrawLabels = val == "true"
		case "VERBOSE_DETECTIONS":
			VerboseDetections = val == "true"
		case "JPEG_QUALITY":
			if n, err := strconv.Atoi(val); err == nil && n > 0 {
				JPEGQuality = n
			}
		case "CLASSES":
			EnvClasses = val
		}
	}
}

const tileSize = 640
const tileStride = 576

type SaveTask struct {
	tile      *image.NRGBA
	boxes     []Detection
	globalIDs []int // 每个 box 在总结文件中的全局序号（1-based）
	outDir    string
	x0, y0    int
	mapInfo   *ENVIMapInfo
}

func main() {
	// 加载顺序：.env < config.json < 命令行参数
	loadEnv("./config.env")
	cfg := loadConfig("./config.json")
	if cfg.Input != "" {
		InputImagePath = cfg.Input
	}
	if cfg.Output != "" {
		OutputImagePath = cfg.Output
	}
	if cfg.Labels != nil {
		DrawLabels = *cfg.Labels
	}
	if cfg.Verbose != nil {
		VerboseDetections = *cfg.Verbose
	}

	// 命令行参数（优先级最高）
	// classes 默认值：config.json > .env
	classesDefault := EnvClasses
	if cfg.Classes != "" {
		classesDefault = cfg.Classes
	}
	classesFlag := flag.String("classes", classesDefault, "要检测的类别，逗号分隔，可用 ID 或名称，例如: 0,1 或 oil,bridge。留空表示检测全部类别。")
	flag.StringVar(&InputImagePath, "input", InputImagePath, "输入 .dat 文件路径")
	flag.StringVar(&OutputImagePath, "output", OutputImagePath, "输出目录/文件路径")
	flag.BoolVar(&DrawLabels, "labels", DrawLabels, "是否在检测框上绘制标签")
	flag.BoolVar(&VerboseDetections, "verbose", VerboseDetections, "是否输出逐框日志")
	flag.BoolVar(&UseCPU, "cpu", UseCPU, "使用 CPU 推理（禁用 GPU/TensorRT，无需 CUDA 环境）")
	flag.Parse()

	// CPU 模式下关闭 GPU 和 TensorRT
	if UseCPU {
		UseGPU = false
		UseTensorRT = false
		fmt.Println("推理模式: CPU（注意：CPU 推理速度较慢，适合无 GPU 环境）")
	} else {
		fmt.Println("推理模式: GPU (TensorRT)")
	}

	// 解析类别过滤
	if *classesFlag != "" {
		EnabledClasses = make(map[int]struct{})
		for _, token := range strings.Split(*classesFlag, ",") {
			token = strings.TrimSpace(token)
			if id, err := strconv.Atoi(token); err == nil {
				EnabledClasses[id] = struct{}{}
			} else {
				// 按名称匹配
				matched := false
				for id, name := range yolo_classes {
					if strings.EqualFold(name, token) {
						EnabledClasses[id] = struct{}{}
						matched = true
						break
					}
				}
				if !matched {
					fmt.Printf("警告: 未知类别 %q，已忽略\n", token)
				}
			}
		}
		// 打印已启用的类别
		enabled := make([]string, 0, len(EnabledClasses))
		for id := range EnabledClasses {
			zh := yolo_classes_zh[yolo_classes[id]]
			enabled = append(enabled, fmt.Sprintf("%s(%s)", yolo_classes[id], zh))
		}
		sort.Strings(enabled)
		fmt.Printf("已启用类别过滤，仅检测: %s\n", strings.Join(enabled, ", "))
	} else {
		fmt.Println("未指定类别过滤，检测全部类别")
	}

	totalStart := time.Now()

	if err := addThirdPartyToDllSearch(); err != nil {
		fmt.Printf("设置 DLL 搜索路径失败: %v\n", err)
		os.Exit(1)
	}

	if err := InitOrtEnv(); err != nil {
		fmt.Printf("ORT 初始化失败: %v\n", err)
		os.Exit(1)
	}

	datPath := InputImagePath
	hdrPath := strings.TrimSuffix(datPath, ".dat") + ".hdr"

	hdr, err := ParseENVIHeader(hdrPath)
	if err != nil {
		fmt.Printf("HDR 解析失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("图像尺寸: %d×%d, %d 波段, 数据类型: %d, 排列: %s\n",
		hdr.Samples, hdr.Lines, hdr.Bands, hdr.DataType, hdr.Interleave)

	reader, err := NewENVITileReader(datPath, hdr)
	if err != nil {
		fmt.Printf("打开数据文件失败: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	outDir := filepath.Dir(OutputImagePath)

	tileCount := 0
	savedCount := 0
	classStats := make(map[int]int) // 各类别检测总数

	// 全局检测记录，用于最终写总结文件
	type DetectionRecord struct {
		Label  string
		Score  float32
		Coord  string
	}
	var (
		allRecords   []DetectionRecord
		allRecordsMu sync.Mutex
	)

	var totalReadTime, totalPrepareTime, totalInferTime, totalPostTime, totalSaveTime time.Duration
	var saveTimeMu sync.Mutex

	// 异步保存 worker pool
	saveChan := make(chan SaveTask, SaveWorkers*2)
	var saveWg sync.WaitGroup

	for i := 0; i < SaveWorkers; i++ {
		saveWg.Add(1)
		go func() {
			defer saveWg.Done()
			for task := range saveChan {
				t4 := time.Now()
				if err := saveTileByClass(task.tile, task.boxes, task.globalIDs, task.outDir, task.x0, task.y0, task.mapInfo); err != nil {
					fmt.Printf("保存瓦片(%d,%d)失败: %v\n", task.x0, task.y0, err)
				} else {
					saveTimeMu.Lock()
					totalSaveTime += time.Since(t4)
					saveTimeMu.Unlock()
				}
			}
		}()
	}

	// 批量推理缓冲区
	type TileBatch struct {
		tile        *image.NRGBA
		input       []float32
		tw, th      int64
		x0, y0      int
		readTime    time.Duration
		prepareTime time.Duration
	}
	type PreparedTile struct {
		tile        *image.NRGBA
		input       []float32
		tw, th      int64
		x0, y0      int
		readTime    time.Duration
		prepareTime time.Duration
		err         error
	}

	preparedChan := make(chan PreparedTile, BatchSize*2)
	go func() {
		defer close(preparedChan)
		for y0 := 0; y0 < hdr.Lines; y0 += tileStride {
			for x0 := 0; x0 < hdr.Samples; x0 += tileStride {
				t0 := time.Now()
				tile, err := reader.ReadTileRGB(x0, y0, tileSize, tileSize)
				readTime := time.Since(t0)
				if err != nil {
					preparedChan <- PreparedTile{x0: x0, y0: y0, err: err, readTime: readTime}
					continue
				}

				t1 := time.Now()
				input, tw, th := prepare_input_image(tile)
				preparedChan <- PreparedTile{
					tile:        tile,
					input:       input,
					tw:          tw,
					th:          th,
					x0:          x0,
					y0:          y0,
					readTime:    readTime,
					prepareTime: time.Since(t1),
				}
			}
		}
	}()

	batch := make([]TileBatch, 0, BatchSize)

	processBatch := func() {
		if len(batch) == 0 {
			return
		}

		// 收集所有输入
		inputs := make([][]float32, len(batch))
		for i := range batch {
			inputs[i] = batch[i].input
		}

		// 批量推理
		t2 := time.Now()
		outputs, err := run_model_batch(inputs)
		totalInferTime += time.Since(t2)
		if err != nil {
			fmt.Printf("批量推理失败: %v\n", err)
			batch = batch[:0]
			return
		}

		// 处理每个瓦片的输出
		for i, tb := range batch {
			t3 := time.Now()
			boxes := process_output(outputs[i], tb.tw, tb.th)
			totalPostTime += time.Since(t3)
			totalReadTime += tb.readTime
			totalPrepareTime += tb.prepareTime

			// 按启用类别过滤
			if EnabledClasses != nil {
				filtered := boxes[:0]
				for _, box := range boxes {
					if _, ok := EnabledClasses[box.ClassID]; ok {
						filtered = append(filtered, box)
					}
				}
				boxes = filtered
			}

			// 统计各类别数量
			for _, box := range boxes {
				classStats[box.ClassID]++
			}

			if len(boxes) > 0 {
				// 记录检测结果，同时生成全局序号
				allRecordsMu.Lock()
				globalIDs := make([]int, len(boxes))
				for j, box := range boxes {
					cx := float64(tb.x0) + float64(box.X1+box.X2)/2
					cy := float64(tb.y0) + float64(box.Y1+box.Y2)/2
					coord := fmt.Sprintf("像素 (%.0f, %.0f)", cx, cy)
					if hdr.MapInfo != nil {
						lat, lon := pixelToLatLon(cx, cy, hdr.MapInfo)
						coord = fmt.Sprintf("%.6f°N  %.6f°E", lat, lon)
					}
					allRecords = append(allRecords, DetectionRecord{
						Label: box.Label(),
						Score: box.Score,
						Coord: coord,
					})
					globalIDs[j] = len(allRecords) // 1-based
				}
				allRecordsMu.Unlock()

				// 异步保存
				saveChan <- SaveTask{
					tile:      tb.tile,
					boxes:     boxes,
					globalIDs: globalIDs,
					outDir:    outDir,
					x0:        tb.x0,
					y0:        tb.y0,
					mapInfo:   hdr.MapInfo,
				}
				savedCount++
				if VerboseDetections {
					for _, box := range boxes {
						fmt.Printf("  [%s] 置信度=%.2f  瓦片偏移=(%d,%d)  坐标=(%.0f,%.0f)-(%.0f,%.0f)\n",
							box.Label(), box.Score, tb.x0, tb.y0,
							box.X1, box.Y1, box.X2, box.Y2)
					}
				}
			}
		}

		batch = batch[:0]
	}

	for prepared := range preparedChan {
		if prepared.err != nil {
			fmt.Printf("读取瓦片(%d,%d)失败: %v\n", prepared.x0, prepared.y0, prepared.err)
			continue
		}

		batch = append(batch, TileBatch{
			tile:        prepared.tile,
			input:       prepared.input,
			tw:          prepared.tw,
			th:          prepared.th,
			x0:          prepared.x0,
			y0:          prepared.y0,
			readTime:    prepared.readTime,
			prepareTime: prepared.prepareTime,
		})

		tileCount++

		if len(batch) >= BatchSize {
			processBatch()
		}

		if tileCount%50 == 0 {
			elapsed := time.Since(totalStart)
			fmt.Printf("已处理 %d 个瓦片, 已保存 %d 个含目标瓦片, 耗时 %v (均速 %.1f ms/瓦片)...\n",
				tileCount, savedCount, elapsed.Round(time.Millisecond),
				float64(elapsed.Milliseconds())/float64(tileCount))
		}
	}

	// 处理剩余的不满批次
	processBatch()

	// 等待所有保存完成
	close(saveChan)
	saveWg.Wait()

	total := time.Since(totalStart)
	fmt.Printf("\n完成: 共处理 %d 个瓦片, 保存 %d 个含目标瓦片到 %s\n", tileCount, savedCount, outDir)
	fmt.Printf("总耗时: %v\n", total.Round(time.Millisecond))

	// 各类别检测统计
	if len(classStats) > 0 {
		fmt.Println("\n各类别检测统计:")
		ids := make([]int, 0, len(classStats))
		for id := range classStats {
			ids = append(ids, id)
		}
		sort.Ints(ids)
		totalDetections := 0
		for _, id := range ids {
			cnt := classStats[id]
			totalDetections += cnt
			zh := yolo_classes_zh[yolo_classes[id]]
			fmt.Printf("  %-20s (%s): %d 个\n", yolo_classes[id], zh, cnt)
		}
		fmt.Printf("  %-20s       : %d 个\n", "合计", totalDetections)
	} else {
		fmt.Println("\n未检测到任何目标")
	}

	// 写总结文件
	if len(allRecords) > 0 {
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("# 检测总结  输入: %s\n", InputImagePath))
		sb.WriteString(fmt.Sprintf("# 共检测到 %d 个目标\n", len(allRecords)))
		sb.WriteString("#\n")
		sb.WriteString(fmt.Sprintf("# %-4s  %-20s  %-6s  %s\n", "序号", "类别", "置信度", "中心点经纬度"))
		for i, r := range allRecords {
			zh := yolo_classes_zh[r.Label]
			sb.WriteString(fmt.Sprintf("  %-4d  %-20s  %.4f  %s\n", i+1, r.Label+"("+zh+")", r.Score, r.Coord))
		}
		summaryPath := filepath.Join(outDir, "detections.txt")
		if err := os.WriteFile(summaryPath, []byte(sb.String()), 0644); err != nil {
			fmt.Printf("写总结文件失败: %v\n", err)
		} else {
			fmt.Printf("\n总结文件已写入: %s\n", summaryPath)
		}
	}
	if tileCount > 0 {
		fmt.Printf("各阶段耗时统计 (共 %d 瓦片):\n", tileCount)
		fmt.Printf("  读取瓦片:   %v (均 %.1f ms)\n", totalReadTime.Round(time.Millisecond), float64(totalReadTime.Milliseconds())/float64(tileCount))
		fmt.Printf("  预处理:     %v (均 %.1f ms)\n", totalPrepareTime.Round(time.Millisecond), float64(totalPrepareTime.Milliseconds())/float64(tileCount))
		fmt.Printf("  模型推理:   %v (均 %.1f ms)\n", totalInferTime.Round(time.Millisecond), float64(totalInferTime.Milliseconds())/float64(tileCount))
		fmt.Printf("  后处理NMS:  %v (均 %.1f ms)\n", totalPostTime.Round(time.Millisecond), float64(totalPostTime.Milliseconds())/float64(tileCount))
		fmt.Printf("  保存图像:   %v (均 %.1f ms/有目标瓦片)\n", totalSaveTime.Round(time.Millisecond),
			func() float64 {
				if savedCount == 0 {
					return 0
				}
				return float64(totalSaveTime.Milliseconds()) / float64(savedCount)
			}())
	}
}

// saveTileByClass 将标注图复制到每个类别子文件夹。
// 每张子图：当前类别目标标注全局序号，其他类别目标标注中文名；右侧面板列出当前类别的序号和经纬度。
func saveTileByClass(tile *image.NRGBA, boxes []Detection, globalIDs []int, outDir string, x0, y0 int, mapInfo *ENVIMapInfo) error {
	// 按类别分组，保留全局序号
	type indexedBox struct {
		box      Detection
		globalID int
	}
	byLabel := make(map[string][]indexedBox)
	for i, box := range boxes {
		byLabel[box.Label()] = append(byLabel[box.Label()], indexedBox{box, globalIDs[i]})
	}

	// 中心点坐标格式化
	centerCoord := func(box Detection) string {
		cx := float64(x0) + float64(box.X1+box.X2)/2
		cy := float64(y0) + float64(box.Y1+box.Y2)/2
		if mapInfo != nil {
			lat, lon := pixelToLatLon(cx, cy, mapInfo)
			return fmt.Sprintf("%.6f°N  %.6f°E", lat, lon)
		}
		return fmt.Sprintf("像素 (%.0f, %.0f)", cx, cy)
	}

	baseName := fmt.Sprintf("tile_%d_%d", x0, y0)
	for label, classBoxes := range byLabel {
		// 为该类别单独画一张图
		annotated := image.NewRGBA(tile.Bounds())
		draw.Draw(annotated, tile.Bounds(), tile, tile.Bounds().Min, draw.Src)

		// 先画所有框（不含标注）
		for _, box := range boxes {
			c := classColor(box.ClassID)
			drawRect(annotated, int(box.X1), int(box.Y1), int(box.X2), int(box.Y2), 3, c)
		}

		// 收集所有标注请求，统一去冲突后绘制
		var labelReqs []LabelRequest
		for i, box := range boxes {
			c := classColor(box.ClassID)
			if box.Label() == label {
				// 当前类别：始终标注全局序号
				labelReqs = append(labelReqs, LabelRequest{int(box.X1), int(box.Y1), int(box.X2 - box.X1), int(box.Y2 - box.Y1), fmt.Sprintf("#%d", globalIDs[i]), c})
			} else if DrawLabels {
				// 其他类别：仅在 DrawLabels 开启时标注中文名
				zh, ok := yolo_classes_zh[box.Label()]
				if !ok {
					zh = box.Label()
				}
				labelReqs = append(labelReqs, LabelRequest{int(box.X1), int(box.Y1), int(box.X2 - box.X1), int(box.Y2 - box.Y1), zh, c})
			}
		}
		DrawLabelsNoOverlap(annotated, labelReqs)

		zh := yolo_classes_zh[label]
		lines := []string{
			fmt.Sprintf("%s (%s)", label, zh),
			"",
			"序号  置信度  中心点经纬度",
		}
		for _, ib := range classBoxes {
			lines = append(lines, fmt.Sprintf("#%-4d  %.2f  %s", ib.globalID, ib.box.Score, centerCoord(ib.box)))
		}

		withPanel := AppendInfoPanel(annotated, lines, classColor(classBoxes[0].box.ClassID), 300)

		subDir := filepath.Join(outDir, label)
		if err := os.MkdirAll(subDir, 0755); err != nil {
			return err
		}
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, withPanel, &jpeg.Options{Quality: JPEGQuality}); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(subDir, baseName+".jpg"), buf.Bytes(), 0644); err != nil {
			return err
		}
	}
	return nil
}

func run_model(input []float32) ([]float32, error) {
	var err error
	if Yolo8Model.Session == nil {
		// 初始化时需要 BatchSize 大小的输入
		batchInput := make([]float32, BatchSize*len(input))
		copy(batchInput, input)
		Yolo8Model, err = InitYolo8Session(batchInput)
		if err != nil {
			return nil, err
		}
	}
	return runInference(Yolo8Model, input)
}

func run_model_batch(inputs [][]float32) ([][]float32, error) {
	var err error
	if Yolo8Model.Session == nil {
		// 初始化时需要 BatchSize 大小的输入
		singleSize := len(inputs[0])
		batchInput := make([]float32, BatchSize*singleSize)
		Yolo8Model, err = InitYolo8Session(batchInput)
		if err != nil {
			return nil, err
		}
	}
	return runBatchInference(Yolo8Model, inputs)
}
