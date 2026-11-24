package main

import (
	"bufio"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"io/fs"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	portFlag = flag.Int("port", 8092, "演示页面端口")

	sourceImageFlag = flag.String(
		"source-image",
		"./fig/preprocessed.png",
		"原始可视化 PNG 路径",
	)
	binaryFlag = flag.String("binary", "./yolov8s", "检测程序路径")
	inputFlag  = flag.String("input", "./data/GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.dat", "检测输入 DAT 路径")
	outputFlag = flag.String("output", "./output_yolov8s_finetune_best_class/output.jpg", "检测输出路径")
	classFlag  = flag.String("classes", "0,1,5", "检测类别")
	cpuFlag    = flag.Bool("cpu", false, "是否使用 CPU")
	labelsFlag = flag.Bool("labels", true, "是否绘制标签")
)

//go:embed web/index.html
var indexHTML string

var tileFilePattern = regexp.MustCompile(`^tile_(\d+)_(\d+)\.jpg$`)
var totalTargetsPattern = regexp.MustCompile(`^# 共检测到 (\d+) 个目标$`)
var detectionLinePattern = regexp.MustCompile(`^\s*(\d+)\s+(.+?)\((.+?)\)\s+([0-9.]+)\s+(.+)$`)

type resultItem struct {
	ClassName  string `json:"className"`
	TileX      int    `json:"tileX"`
	TileY      int    `json:"tileY"`
	TileW      int    `json:"tileW"`
	TileH      int    `json:"tileH"`
	Relative   string `json:"relative"`
	ImageURL   string `json:"imageURL"`
	DetectedAt string `json:"detectedAt"`
}

type tileOverlay struct {
	TileX   int      `json:"tileX"`
	TileY   int      `json:"tileY"`
	TileW   int      `json:"tileW"`
	TileH   int      `json:"tileH"`
	Classes []string `json:"classes"`
}

type logEntry struct {
	At      string `json:"at"`
	Source  string `json:"source"`
	Message string `json:"message"`
}

type runConfig struct {
	BinaryPath string   `json:"binaryPath"`
	InputPath  string   `json:"inputPath"`
	OutputPath string   `json:"outputPath"`
	Classes    []string `json:"classes"`
	UseCPU     bool     `json:"useCPU"`
	Labels     bool     `json:"labels"`
}

type summaryClassStat struct {
	Name    string `json:"name"`
	ZhName  string `json:"zhName"`
	Count   int    `json:"count"`
	Display string `json:"display"`
}

type summaryDetection struct {
	Index   int     `json:"index"`
	Name    string  `json:"name"`
	ZhName  string  `json:"zhName"`
	Display string  `json:"display"`
	Score   float64 `json:"score"`
	Coord   string  `json:"coord"`
}

type detectionSummary struct {
	Available     bool               `json:"available"`
	Path          string             `json:"path"`
	UpdatedAt     string             `json:"updatedAt"`
	TotalTargets  int                `json:"totalTargets"`
	ClassCount    int                `json:"classCount"`
	ClassStats    []summaryClassStat `json:"classStats"`
	TopDetections []summaryDetection `json:"topDetections"`
}

type state struct {
	Running         bool             `json:"running"`
	Status          string           `json:"status"`
	Error           string           `json:"error"`
	Command         string           `json:"command"`
	StartedAt       string           `json:"startedAt"`
	FinishedAt      string           `json:"finishedAt"`
	ElapsedSeconds  int              `json:"elapsedSeconds"`
	SourceImageName string           `json:"sourceImageName"`
	SourceImageURL  string           `json:"sourceImageURL"`
	ImageWidth      int              `json:"imageWidth"`
	ImageHeight     int              `json:"imageHeight"`
	OutputDir       string           `json:"outputDir"`
	ResultCount     int              `json:"resultCount"`
	TileCount       int              `json:"tileCount"`
	Config          runConfig        `json:"config"`
	Summary         detectionSummary `json:"summary"`
	Results         []resultItem     `json:"results"`
	Tiles           []tileOverlay    `json:"tiles"`
	Logs            []logEntry       `json:"logs"`
}

type hub struct {
	mu   sync.Mutex
	subs map[int]chan []byte
	next int
}

func newHub() *hub {
	return &hub{subs: make(map[int]chan []byte)}
}

func (h *hub) subscribe() (int, chan []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	id := h.next
	h.next++
	ch := make(chan []byte, 8)
	h.subs[id] = ch
	return id, ch
}

func (h *hub) unsubscribe(id int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if ch, ok := h.subs[id]; ok {
		delete(h.subs, id)
		close(ch)
	}
}

func (h *hub) broadcast(v any) {
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, ch := range h.subs {
		select {
		case ch <- data:
		default:
		}
	}
}

type app struct {
	mu      sync.RWMutex
	state   state
	hub     *hub
	cmd     *exec.Cmd
	runFrom time.Time
	seen    map[string]struct{}
}

func newApp() (*app, error) {
	cfg, err := readImageConfig(*sourceImageFlag)
	if err != nil {
		return nil, err
	}
	runCfg := defaultRunConfig()
	return &app{
		state: state{
			Status:          "待启动",
			Command:         buildCommandText(runCfg),
			SourceImageName: filepath.Base(*sourceImageFlag),
			SourceImageURL:  "/api/source",
			ImageWidth:      cfg.Width,
			ImageHeight:     cfg.Height,
			OutputDir:       filepath.Dir(runCfg.OutputPath),
			Config:          runCfg,
			Logs:            []logEntry{},
			Results:         []resultItem{},
			Tiles:           []tileOverlay{},
		},
		hub:  newHub(),
		seen: make(map[string]struct{}),
	}, nil
}

func readImageConfig(path string) (image.Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return image.Config{}, err
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	return cfg, err
}

func defaultRunConfig() runConfig {
	return runConfig{
		BinaryPath: *binaryFlag,
		InputPath:  *inputFlag,
		OutputPath: *outputFlag,
		Classes:    normalizeClasses(strings.Split(*classFlag, ",")),
		UseCPU:     *cpuFlag,
		Labels:     *labelsFlag,
	}
}

func normalizeClasses(raw []string) []string {
	out := make([]string, 0, len(raw))
	seen := make(map[string]struct{})
	for _, item := range raw {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if _, ok := seen[item]; ok {
			continue
		}
		seen[item] = struct{}{}
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool {
		ai, aErr := strconv.Atoi(out[i])
		aj, bErr := strconv.Atoi(out[j])
		if aErr == nil && bErr == nil {
			return ai < aj
		}
		return out[i] < out[j]
	})
	return out
}

func buildCommandArgs(cfg runConfig) []string {
	args := make([]string, 0, 10)
	if cfg.UseCPU {
		args = append(args, "-cpu")
	}
	if len(cfg.Classes) > 0 {
		args = append(args, "-classes", strings.Join(cfg.Classes, ","))
	}
	if cfg.Labels {
		args = append(args, "-labels")
	}
	args = append(args, "-input", cfg.InputPath, "-output", cfg.OutputPath)
	return args
}

func buildCommandText(cfg runConfig) string {
	parts := []string{cfg.BinaryPath}
	for _, arg := range buildCommandArgs(cfg) {
		if strings.ContainsAny(arg, " \t\"") {
			parts = append(parts, fmt.Sprintf("%q", arg))
			continue
		}
		parts = append(parts, arg)
	}
	return strings.Join(parts, " ")
}

func (a *app) snapshot() state {
	a.mu.RLock()
	defer a.mu.RUnlock()
	cp := a.state
	cp.Results = append([]resultItem(nil), a.state.Results...)
	cp.Tiles = append([]tileOverlay(nil), a.state.Tiles...)
	cp.Logs = append([]logEntry(nil), a.state.Logs...)
	cp.Summary.ClassStats = append([]summaryClassStat(nil), a.state.Summary.ClassStats...)
	cp.Summary.TopDetections = append([]summaryDetection(nil), a.state.Summary.TopDetections...)
	return cp
}

func (a *app) broadcast() {
	a.hub.broadcast(map[string]any{
		"type":  "state",
		"state": a.snapshot(),
	})
}

func (a *app) appendLog(source, message string) {
	a.mu.Lock()
	a.state.Logs = append(a.state.Logs, logEntry{
		At:      time.Now().Format("15:04:05"),
		Source:  source,
		Message: message,
	})
	if len(a.state.Logs) > 180 {
		a.state.Logs = append([]logEntry(nil), a.state.Logs[len(a.state.Logs)-180:]...)
	}
	a.mu.Unlock()
	a.broadcast()
}

func (a *app) setRunning(running bool, status string, err error) {
	a.mu.Lock()
	a.state.Running = running
	a.state.Status = status
	if err != nil {
		a.state.Error = err.Error()
	} else {
		a.state.Error = ""
	}
	if !running {
		a.state.FinishedAt = time.Now().Format(time.RFC3339)
	}
	a.mu.Unlock()
	a.broadcast()
}

func (a *app) start(cfg runConfig) error {
	a.mu.Lock()
	if a.state.Running {
		a.mu.Unlock()
		return fmt.Errorf("检测任务正在运行")
	}

	a.runFrom = time.Now()
	a.seen = make(map[string]struct{})
	a.state.Running = true
	a.state.Status = "检测中"
	a.state.Error = ""
	a.state.StartedAt = a.runFrom.Format(time.RFC3339)
	a.state.FinishedAt = ""
	a.state.ElapsedSeconds = 0
	a.state.Results = nil
	a.state.Tiles = nil
	a.state.Logs = nil
	a.state.ResultCount = 0
	a.state.TileCount = 0
	a.state.Summary = detectionSummary{}
	a.state.Config = cfg
	a.state.Command = buildCommandText(cfg)
	a.state.OutputDir = filepath.Dir(cfg.OutputPath)

	cmd := exec.Command(cfg.BinaryPath, buildCommandArgs(cfg)...)
	cmd.Dir = mustGetwd()
	a.cmd = cmd
	a.mu.Unlock()
	a.broadcast()

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		a.setRunning(false, "启动失败", err)
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		a.setRunning(false, "启动失败", err)
		return err
	}
	if err := cmd.Start(); err != nil {
		a.setRunning(false, "启动失败", err)
		return err
	}

	a.appendLog("cmd", "检测命令已启动: "+buildCommandText(cfg))
	go a.scanPipe("stdout", stdout)
	go a.scanPipe("stderr", stderr)
	go a.watchResults(cmd)
	return nil
}

func mustGetwd() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return wd
}

func (a *app) scanPipe(source string, pipe io.Reader) {
	scanner := bufio.NewScanner(pipe)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		a.appendLog(source, line)
	}
}

func (a *app) watchResults(cmd *exec.Cmd) {
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	ticker := time.NewTicker(900 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case err := <-done:
			a.refreshResults(true)
			if err != nil {
				a.appendLog("sys", fmt.Sprintf("检测命令退出失败: %v", err))
				a.setRunning(false, "运行失败", err)
				return
			}
			a.appendLog("sys", "检测命令执行完成")
			a.setRunning(false, "检测完成", nil)
			return
		case <-ticker.C:
			a.refreshResults(false)
		}
	}
}

func (a *app) refreshResults(force bool) {
	a.mu.RLock()
	imgW := a.state.ImageWidth
	imgH := a.state.ImageHeight
	start := a.runFrom
	outputDir := a.state.OutputDir
	seen := make(map[string]struct{}, len(a.seen))
	for k := range a.seen {
		seen[k] = struct{}{}
	}
	a.mu.RUnlock()

	results, tiles, newPaths, err := scanOutputDir(outputDir, start, imgW, imgH, seen)
	if err != nil {
		if force {
			a.appendLog("sys", fmt.Sprintf("扫描输出目录失败: %v", err))
		}
		return
	}

	summary, err := parseDetectionsSummary(filepath.Join(outputDir, "detections.txt"), start)
	if err != nil && force {
		a.appendLog("sys", fmt.Sprintf("解析 detections.txt 失败: %v", err))
	}

	a.mu.Lock()
	a.state.Results = results
	a.state.Tiles = tiles
	a.state.ResultCount = len(results)
	a.state.TileCount = len(tiles)
	a.state.ElapsedSeconds = int(time.Since(a.runFrom).Seconds())
	if err == nil {
		a.state.Summary = summary
	}
	for _, path := range newPaths {
		a.seen[path] = struct{}{}
	}
	a.mu.Unlock()
	a.broadcast()

	for _, path := range newPaths {
		a.appendLog("file", fmt.Sprintf("发现新结果图: %s", path))
	}
}

func scanOutputDir(root string, start time.Time, imgW, imgH int, seen map[string]struct{}) ([]resultItem, []tileOverlay, []string, error) {
	if _, err := os.Stat(root); err != nil {
		if os.IsNotExist(err) {
			return nil, nil, nil, nil
		}
		return nil, nil, nil, err
	}

	type diskItem struct {
		resultItem
		modTime time.Time
	}

	items := make([]diskItem, 0, 32)
	tileMap := make(map[string]*tileOverlay)
	newPaths := make([]string, 0, 16)

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return err
		}
		if info.ModTime().Before(start) {
			return nil
		}

		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		className := filepath.Base(filepath.Dir(path))
		base := filepath.Base(path)
		matches := tileFilePattern.FindStringSubmatch(base)
		if len(matches) != 3 {
			return nil
		}

		x := parseInt(matches[1])
		y := parseInt(matches[2])
		w := minInt(640, maxInt(0, imgW-x))
		h := minInt(640, maxInt(0, imgH-y))

		item := diskItem{
			resultItem: resultItem{
				ClassName:  className,
				TileX:      x,
				TileY:      y,
				TileW:      w,
				TileH:      h,
				Relative:   rel,
				ImageURL:   "/api/result?path=" + url.QueryEscape(rel),
				DetectedAt: info.ModTime().Format("15:04:05"),
			},
			modTime: info.ModTime(),
		}
		items = append(items, item)

		key := fmt.Sprintf("%d_%d", x, y)
		overlay := tileMap[key]
		if overlay == nil {
			overlay = &tileOverlay{TileX: x, TileY: y, TileW: w, TileH: h}
			tileMap[key] = overlay
		}
		if !contains(overlay.Classes, className) {
			overlay.Classes = append(overlay.Classes, className)
			sort.Strings(overlay.Classes)
		}

		if _, ok := seen[rel]; !ok {
			newPaths = append(newPaths, rel)
		}
		return nil
	})
	if err != nil {
		return nil, nil, nil, err
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].modTime.After(items[j].modTime)
	})
	results := make([]resultItem, 0, len(items))
	for _, item := range items {
		results = append(results, item.resultItem)
	}

	tiles := make([]tileOverlay, 0, len(tileMap))
	for _, tile := range tileMap {
		tiles = append(tiles, *tile)
	}
	sort.Slice(tiles, func(i, j int) bool {
		if tiles[i].TileY == tiles[j].TileY {
			return tiles[i].TileX < tiles[j].TileX
		}
		return tiles[i].TileY < tiles[j].TileY
	})
	sort.Strings(newPaths)

	return results, tiles, newPaths, nil
}

func parseDetectionsSummary(path string, start time.Time) (detectionSummary, error) {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return detectionSummary{}, nil
		}
		return detectionSummary{}, err
	}
	if !start.IsZero() && info.ModTime().Before(start) {
		return detectionSummary{}, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return detectionSummary{}, err
	}
	defer f.Close()

	type aggClass struct {
		Name   string
		ZhName string
		Count  int
	}

	var (
		total      int
		classMap   = make(map[string]*aggClass)
		detections []summaryDetection
	)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if m := totalTargetsPattern.FindStringSubmatch(line); len(m) == 2 {
			total, _ = strconv.Atoi(m[1])
			continue
		}

		if strings.HasPrefix(line, "#") {
			continue
		}

		m := detectionLinePattern.FindStringSubmatch(line)
		if len(m) != 6 {
			continue
		}

		index, _ := strconv.Atoi(m[1])
		name := strings.TrimSpace(m[2])
		zhName := strings.TrimSpace(m[3])
		score, _ := strconv.ParseFloat(m[4], 64)
		coord := strings.TrimSpace(m[5])

		key := name + "|" + zhName
		if classMap[key] == nil {
			classMap[key] = &aggClass{Name: name, ZhName: zhName}
		}
		classMap[key].Count++

		detections = append(detections, summaryDetection{
			Index:   index,
			Name:    name,
			ZhName:  zhName,
			Display: zhName + " / " + name,
			Score:   score,
			Coord:   coord,
		})
	}
	if err := scanner.Err(); err != nil {
		return detectionSummary{}, err
	}

	classStats := make([]summaryClassStat, 0, len(classMap))
	for _, item := range classMap {
		classStats = append(classStats, summaryClassStat{
			Name:    item.Name,
			ZhName:  item.ZhName,
			Count:   item.Count,
			Display: item.ZhName + " / " + item.Name,
		})
	}
	sort.Slice(classStats, func(i, j int) bool {
		if classStats[i].Count == classStats[j].Count {
			return classStats[i].Display < classStats[j].Display
		}
		return classStats[i].Count > classStats[j].Count
	})

	sort.Slice(detections, func(i, j int) bool {
		if detections[i].Score == detections[j].Score {
			return detections[i].Index < detections[j].Index
		}
		return detections[i].Score > detections[j].Score
	})
	if len(detections) > 8 {
		detections = detections[:8]
	}
	if total == 0 {
		for _, item := range classStats {
			total += item.Count
		}
	}

	return detectionSummary{
		Available:     true,
		Path:          path,
		UpdatedAt:     info.ModTime().Format("2006-01-02 15:04:05"),
		TotalTargets:  total,
		ClassCount:    len(classStats),
		ClassStats:    classStats,
		TopDetections: detections,
	}, nil
}

func parseStartRequest(r *http.Request, base runConfig) (runConfig, error) {
	cfg := base

	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			return cfg, err
		}
	} else if err := r.ParseForm(); err != nil {
		return cfg, err
	}

	if v := strings.TrimSpace(r.FormValue("binaryPath")); v != "" {
		cfg.BinaryPath = v
	}
	if v := strings.TrimSpace(r.FormValue("inputPath")); v != "" {
		cfg.InputPath = v
	}
	if v := strings.TrimSpace(r.FormValue("outputPath")); v != "" {
		cfg.OutputPath = v
	}

	cfg.UseCPU = parseBoolValue(r.FormValue("useCPU"))
	cfg.Labels = parseBoolValue(r.FormValue("labels"))

	classes := normalizeClasses(r.Form["classes"])
	if len(classes) == 0 {
		if raw := strings.TrimSpace(r.FormValue("classes")); raw != "" {
			classes = normalizeClasses(strings.Split(raw, ","))
		}
	}
	cfg.Classes = classes

	uploadedDatPath, err := saveUploadedInputFiles(r)
	if err != nil {
		return cfg, err
	}
	if uploadedDatPath != "" {
		cfg.InputPath = uploadedDatPath
	}

	if cfg.BinaryPath == "" {
		return cfg, fmt.Errorf("检测程序路径不能为空")
	}
	if cfg.InputPath == "" {
		return cfg, fmt.Errorf("请输入 DAT 路径或上传 DAT/HDR 文件")
	}
	if cfg.OutputPath == "" {
		return cfg, fmt.Errorf("输出路径不能为空")
	}

	if strings.EqualFold(filepath.Ext(cfg.InputPath), ".dat") {
		hdrPath := strings.TrimSuffix(cfg.InputPath, filepath.Ext(cfg.InputPath)) + ".hdr"
		if _, err := os.Stat(hdrPath); err != nil {
			return cfg, fmt.Errorf("未找到对应 HDR 文件: %s", hdrPath)
		}
	}

	return cfg, nil
}

func parseBoolValue(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}

func saveUploadedInputFiles(r *http.Request) (string, error) {
	if r.MultipartForm == nil {
		return "", nil
	}

	datFile, datHeader, err := r.FormFile("inputDatFile")
	if err != nil {
		if err == http.ErrMissingFile {
			return "", nil
		}
		return "", err
	}
	defer datFile.Close()

	hdrFile, hdrHeader, err := r.FormFile("inputHdrFile")
	if err != nil {
		if err == http.ErrMissingFile {
			return "", fmt.Errorf("上传 DAT 文件时需要同时上传对应 HDR 文件")
		}
		return "", err
	}
	defer hdrFile.Close()

	uploadDir := filepath.Join("demo_uploads", time.Now().Format("20060102_150405"))
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		return "", err
	}

	datPath, err := writeUploadedFile(datFile, datHeader, uploadDir)
	if err != nil {
		return "", err
	}
	if _, err := writeUploadedFile(hdrFile, hdrHeader, uploadDir); err != nil {
		return "", err
	}
	return datPath, nil
}

func writeUploadedFile(src multipart.File, header *multipart.FileHeader, dir string) (string, error) {
	name := filepath.Base(header.Filename)
	if name == "." || name == string(filepath.Separator) || name == "" {
		return "", fmt.Errorf("非法上传文件名")
	}

	dstPath := filepath.Join(dir, name)
	dst, err := os.Create(dstPath)
	if err != nil {
		return "", err
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		return "", err
	}
	return dstPath, nil
}

func contains(items []string, needle string) bool {
	for _, item := range items {
		if item == needle {
			return true
		}
	}
	return false
}

func parseInt(s string) int {
	n := 0
	for _, ch := range s {
		n = n*10 + int(ch-'0')
	}
	return n
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (a *app) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(indexHTML))
}

func (a *app) handleState(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, a.snapshot())
}

func (a *app) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	cfg, err := parseStartRequest(r, a.snapshot().Config)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if err := a.start(cfg); err != nil {
		writeJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"status": "started"})
}

func (a *app) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "stream unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id, ch := a.hub.subscribe()
	defer a.hub.unsubscribe(id)

	initial, _ := json.Marshal(map[string]any{"type": "state", "state": a.snapshot()})
	_, _ = fmt.Fprintf(w, "data: %s\n\n", initial)
	flusher.Flush()

	keepalive := time.NewTicker(25 * time.Second)
	defer keepalive.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case data, ok := <-ch:
			if !ok {
				return
			}
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		case <-keepalive.C:
			_, _ = fmt.Fprint(w, ": ping\n\n")
			flusher.Flush()
		}
	}
}

func (a *app) handleSource(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, *sourceImageFlag)
}

func (a *app) handleResult(w http.ResponseWriter, r *http.Request) {
	rel := r.URL.Query().Get("path")
	if rel == "" {
		http.NotFound(w, r)
		return
	}
	rel = filepath.Clean(filepath.FromSlash(rel))
	if strings.HasPrefix(rel, "..") {
		http.NotFound(w, r)
		return
	}
	a.mu.RLock()
	outputDir := a.state.OutputDir
	a.mu.RUnlock()
	http.ServeFile(w, r, filepath.Join(outputDir, rel))
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func main() {
	flag.Parse()

	app, err := newApp()
	if err != nil {
		fmt.Fprintf(os.Stderr, "初始化演示页面失败: %v\n", err)
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", app.handleIndex)
	mux.HandleFunc("/api/state", app.handleState)
	mux.HandleFunc("/api/start", app.handleStart)
	mux.HandleFunc("/api/events", app.handleEvents)
	mux.HandleFunc("/api/source", app.handleSource)
	mux.HandleFunc("/api/result", app.handleResult)

	addr := fmt.Sprintf(":%d", *portFlag)
	fmt.Printf("演示页面已启动: http://127.0.0.1%s\n", addr)
	fmt.Printf("检测命令: %s\n", buildCommandText(defaultRunConfig()))
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintf(os.Stderr, "演示页面服务失败: %v\n", err)
		os.Exit(1)
	}
}
