package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"math"
	"os"
	"sort"
	"sync"

	"github.com/nfnt/resize"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// classColors 按类别 ID 分配不同颜色，循环使用。
var classColors = []color.RGBA{
	{0, 255, 0, 255},   // 绿
	{255, 0, 0, 255},   // 红
	{0, 0, 255, 255},   // 蓝
	{255, 255, 0, 255}, // 黄
	{255, 0, 255, 255}, // 品红
	{0, 255, 255, 255}, // 青
	{255, 128, 0, 255}, // 橙
	{128, 0, 255, 255}, // 紫
}

func classColor(classID int) color.RGBA {
	return classColors[classID%len(classColors)]
}


// Lower threshold → more detections (fewer misses).
// Higher threshold → fewer detections (fewer false positives).
var classConfThresh = map[int]float32{
	0: 0.50, // oil
	1: 0.50, // bridge
	2: 0.50, // basketball court
	3: 0.55, // ground track field — 容易误检为 roundabout，略提高
	4: 0.55, // roundabout         — 同上
	5: 0.45, // harbor             — 标注少，保持低阈值
	6: 0.50, // other court
	7: 0.65, // soccer ball field  — 误检水体/农田，提高阈值
}

type Detection struct {
	X1      float32
	Y1      float32
	X2      float32
	Y2      float32
	ClassID int
	Score   float32
}

func (d Detection) Label() string {
	return yolo_classes[d.ClassID]
}

// Function used to convert RAW output from YOLOv8-OBB to detected boxes.
// Output shape is (1, 13, 8400):
//   rows 0-3:   xywh box coords (normalized to 640 space)
//   rows 4-11:  class confidence scores (8 classes)
//   row  12:    rotation angle in radians
func process_output(output []float32, img_width, img_height int64) []Detection {
	const numAnchors = 8400
	const numClasses = 8

	boxes := make([]Detection, 0, 64)
	for index := 0; index < numAnchors; index++ {
		classID, prob := 0, float32(0.0)
		for col := 0; col < numClasses; col++ {
			score := output[numAnchors*(col+4)+index]
			if score > prob {
				prob = score
				classID = col
			}
		}
		if prob < classConfThresh[classID] {
			continue
		}
		xc := output[index]
		yc := output[numAnchors+index]
		w := output[2*numAnchors+index]
		h := output[3*numAnchors+index]
		boxes = append(boxes, Detection{
			X1:      (xc - w/2) / 640 * float32(img_width),
			Y1:      (yc - h/2) / 640 * float32(img_height),
			X2:      (xc + w/2) / 640 * float32(img_width),
			Y2:      (yc + h/2) / 640 * float32(img_height),
			ClassID: classID,
			Score:   prob,
		})
	}

	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Score > boxes[j].Score
	})
	result := make([]Detection, 0, len(boxes))
	for len(boxes) > 0 {
		best := boxes[0]
		result = append(result, best)
		tmp := boxes[:0]
		for _, box := range boxes[1:] {
			if iou(best, box) < 0.45 {
				tmp = append(tmp, box)
			}
		}
		boxes = tmp
	}
	return result
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes.
func iou(box1, box2 Detection) float64 {
	return intersection(box1, box2) / union(box1, box2)
}

// Function calculates union area of two boxes.
func union(box1, box2 Detection) float64 {
	box1Area := float64(box1.X2-box1.X1) * float64(box1.Y2-box1.Y1)
	box2Area := float64(box2.X2-box2.X1) * float64(box2.Y2-box2.Y1)
	return box1Area + box2Area - intersection(box1, box2)
}

// Function calculates intersection area of two boxes.
func intersection(box1, box2 Detection) float64 {
	x1 := math.Max(float64(box1.X1), float64(box2.X1))
	y1 := math.Max(float64(box1.Y1), float64(box2.Y1))
	x2 := math.Min(float64(box1.X2), float64(box2.X2))
	y2 := math.Min(float64(box1.Y2), float64(box2.Y2))
	if x2 <= x1 || y2 <= y1 {
		return 0
	}
	return (x2 - x1) * (y2 - y1)
}

// Function used to convert input image to tensor,
// required as an input to YOLOv8 object detection
// network.
// Returns the input tensor, original image width and height
func prepare_input(buf io.Reader) ([]float32, int64, int64) {
	img, _, err := image.Decode(buf)
	if err != nil {
		panic(fmt.Sprintf("图像解码失败: %v", err))
	}
	return prepare_input_image(img)
}

// prepare_input_image converts an image.Image to a YOLOv8 input tensor.
func prepare_input_image(img image.Image) ([]float32, int64, int64) {
	size := img.Bounds().Size()
	img_width, img_height := int64(size.X), int64(size.Y)

	// Only resize if the tile is not already 640×640 (edge tiles).
	var src *image.NRGBA
	if size.X == 640 && size.Y == 640 {
		if n, ok := img.(*image.NRGBA); ok {
			src = n
		}
	}
	if src == nil {
		resized := resize.Resize(640, 640, img, resize.Bilinear)
		if n, ok := resized.(*image.NRGBA); ok {
			src = n
		} else {
			// Convert to NRGBA
			src = image.NewNRGBA(image.Rect(0, 0, 640, 640))
			draw.Draw(src, src.Bounds(), resized, resized.Bounds().Min, draw.Src)
		}
	}

	input := make([]float32, 3*640*640)
	rOff := 0
	gOff := 640 * 640
	bOff := 2 * 640 * 640
	pix := src.Pix // NRGBA stride is 4 bytes: R,G,B,A
	for i := 0; i < 640*640; i++ {
		input[rOff+i] = float32(pix[i*4]) / 255.0
		input[gOff+i] = float32(pix[i*4+1]) / 255.0
		input[bOff+i] = float32(pix[i*4+2]) / 255.0
	}
	return input, img_width, img_height
}

// draw_and_save draws bounding boxes on src and saves as JPEG to outPath.
func draw_and_save(src image.Image, boxes []Detection, outPath string) error {
	// Convert to RGBA so we can draw on it
	bounds := src.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, src, bounds.Min, draw.Src)

	green := color.RGBA{0, 255, 0, 255}
	const thickness = 2

	for _, box := range boxes {
		x1 := int(box.X1)
		y1 := int(box.Y1)
		x2 := int(box.X2)
		y2 := int(box.Y2)
		drawRect(dst, x1, y1, x2, y2, thickness, green)
	}

	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	return jpeg.Encode(out, dst, &jpeg.Options{Quality: 95})
}

// drawRect draws a rectangle border with the given thickness and color.
func drawRect(img *image.RGBA, x1, y1, x2, y2, thickness int, c color.RGBA) {
	for t := 0; t < thickness; t++ {
		// top & bottom edges
		for x := x1; x <= x2; x++ {
			img.SetRGBA(x, y1+t, c)
			img.SetRGBA(x, y2-t, c)
		}
		// left & right edges
		for y := y1; y <= y2; y++ {
			img.SetRGBA(x1+t, y, c)
			img.SetRGBA(x2-t, y, c)
		}
	}
}

// --- 字体加载（懒加载，仅初始化一次） ---

var (
	labelFace     font.Face
	labelFaceOnce sync.Once
	labelFaceMu   sync.Mutex // 保护并发访问
)

// 候选中文字体路径
var cjkFontPaths = []string{
	"/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
	"/usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf",
	"/usr/share/fonts/truetype/arphic/uming.ttc",
	"/usr/share/fonts/truetype/arphic/ukai.ttc",
}

func getLabelFace() font.Face {
	labelFaceOnce.Do(func() {
		for _, p := range cjkFontPaths {
			data, err := os.ReadFile(p)
			if err != nil {
				continue
			}
			col, err := opentype.ParseCollection(data)
			if err != nil {
				// 尝试单字体
				ft, err2 := opentype.Parse(data)
				if err2 != nil {
					continue
				}
				face, err2 := opentype.NewFace(ft, &opentype.FaceOptions{
					Size: 12, DPI: 96,
				})
				if err2 == nil {
					labelFace = face
					return
				}
				continue
			}
			ft, err := col.Font(0)
			if err != nil {
				continue
			}
			face, err := opentype.NewFace(ft, &opentype.FaceOptions{
				Size: 12, DPI: 96,
			})
			if err == nil {
				labelFace = face
				return
			}
		}
	})
	return labelFace
}

// LabelRequest 描述一个待绘制的标注。
type LabelRequest struct {
	X, Y        int // 检测框左上角像素坐标
	BoxW, BoxH  int // 检测框宽高，用于限制偏移范围
	Text        string
	Color       color.RGBA
}

// DrawLabelsNoOverlap 批量绘制标注，自动消除重叠。
// 按优先级尝试四条边（上→左→右→下），每条边紧贴检测框；
// 四边均被占用时缩小字号重试；最终兜底画在框内左上角。
func DrawLabelsNoOverlap(img *image.RGBA, reqs []LabelRequest) {
	labelFaceMu.Lock()
	defer labelFaceMu.Unlock()

	face := getLabelFace()
	if face == nil {
		return
	}
	metrics := face.Metrics()
	lineH := metrics.Height.Ceil() + 2
	bounds := img.Bounds()

	type slot struct{ x, y, w, h int }
	occupied := make([]slot, 0, len(reqs))

	// y 是基线坐标（文字底部），矩形占用区域为 [x, y-h, x+w, y]
	overlaps := func(x, y, w, h int) bool {
		for _, p := range occupied {
			if x < p.x+p.w && x+w > p.x && y-h < p.y && y > p.y-p.h {
				return true
			}
		}
		return false
	}

	clamp := func(tx, ty, textW, lh int) (int, int) {
		if tx+textW > bounds.Max.X {
			tx = bounds.Max.X - textW - 1
		}
		if tx < bounds.Min.X {
			tx = bounds.Min.X + 1
		}
		if ty > bounds.Max.Y {
			ty = bounds.Max.Y - 1
		}
		if ty-lh < bounds.Min.Y {
			ty = bounds.Min.Y + lh
		}
		return tx, ty
	}

	for _, req := range reqs {
		nRunes := len([]rune(req.Text))

		// 尝试不同字符宽度（正常→缩小），每种宽度都试四条边
		found := false
		var finalX, finalY, finalW int
		for _, charW := range []int{10, 7, 5} {
			textW := nRunes * charW
			// 限制文字宽度不超过框的较长边，避免过宽
			maxW := req.BoxW
			if req.BoxH > maxW {
				maxW = req.BoxH
			}
			if textW > maxW && maxW > 0 {
				textW = maxW
			}

			// 四条边候选位置：上、下、左、右
			// ty 为基线（文字底部）
			candidates := []struct{ tx, ty int }{
				// 上边：文字底部贴框顶
				{req.X + 2, req.Y},
				// 下边：文字顶部贴框底
				{req.X + 2, req.Y + req.BoxH + lineH},
				// 左边：文字右侧贴框左
				{req.X - textW - 2, req.Y + lineH},
				// 右边：文字左侧贴框右
				{req.X + req.BoxW + 2, req.Y + lineH},
			}

			for _, c := range candidates {
				tx, ty := clamp(c.tx, c.ty, textW, lineH)
				if !overlaps(tx, ty, textW, lineH) {
					finalX, finalY, finalW = tx, ty, textW
					found = true
					break
				}
			}
			if found {
				break
			}
		}

		// 兜底：强制画在框内左上角，不再漂移
		if !found {
			textW := nRunes * 5
			finalX = req.X + 2
			finalY = req.Y + lineH + 1
			finalX, finalY = clamp(finalX, finalY, textW, lineH)
			finalW = textW
		}

		drawTextLine(img, finalX, finalY, req.Text, req.Color)
		occupied = append(occupied, slot{finalX, finalY, finalW, lineH})
	}
}

// drawLabel 在检测框左上角绘制标签，自动将位置限制在图像范围内。
// 单次调用不做跨标注去冲突，多目标场景请用 DrawLabelsNoOverlap。
func drawLabel(img *image.RGBA, x, y int, text string, c color.RGBA) {
	labelFaceMu.Lock()
	defer labelFaceMu.Unlock()

	face := getLabelFace()
	if face == nil {
		return
	}
	metrics := face.Metrics()
	lineH := metrics.Height.Ceil()

	bounds := img.Bounds()

	// 估算文字宽度（每字符约 10px，粗略保证不超右边界）
	textW := len([]rune(text)) * 10

	// X：优先从框左侧开始，超出右边界则右对齐
	tx := x + 2
	if tx+textW > bounds.Max.X {
		tx = bounds.Max.X - textW - 2
	}
	if tx < bounds.Min.X {
		tx = bounds.Min.X + 2
	}

	// Y：优先放在框上方，放不下则放框内侧下移一行
	ty := y
	if ty-lineH < bounds.Min.Y {
		ty = y + lineH + 2
	}
	if ty > bounds.Max.Y {
		ty = bounds.Max.Y - 2
	}

	drawTextLine(img, tx, ty, text, c)
}

// drawTextLine 在指定坐标绘制一行文字，不加锁，调用方负责加锁。
// 若字体未加载则跳过。
func drawTextLine(img *image.RGBA, x, y int, text string, c color.RGBA) {
	face := getLabelFace()
	if face == nil {
		return
	}
	metrics := face.Metrics()
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: face,
		Dot:  fixed.P(x, y-metrics.Descent.Ceil()-1),
	}
	d.DrawString(text)
}

// AppendInfoPanel 在 src 右侧拼接一块白色面板，写入 lines 文字，返回新图像。
// lineColor 为文字颜色，panelWidth 为面板宽度（像素）。
func AppendInfoPanel(src *image.RGBA, lines []string, lineColor color.RGBA, panelWidth int) *image.RGBA {
	labelFaceMu.Lock()
	defer labelFaceMu.Unlock()

	face := getLabelFace()
	lineHeight := 20
	if face != nil {
		m := face.Metrics()
		lineHeight = m.Height.Ceil() + 4
	}

	srcB := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, srcB.Max.X+panelWidth, srcB.Max.Y))

	// 复制原图
	draw.Draw(dst, srcB, src, srcB.Min, draw.Src)

	// 白色背景面板
	panelRect := image.Rect(srcB.Max.X, 0, srcB.Max.X+panelWidth, srcB.Max.Y)
	draw.Draw(dst, panelRect, image.White, image.Point{}, draw.Src)

	// 写文字
	if face != nil {
		x := srcB.Max.X + 8
		y := lineHeight
		for _, line := range lines {
			drawTextLine(dst, x, y, line, lineColor)
			y += lineHeight
			if y+lineHeight > srcB.Max.Y {
				break
			}
		}
	}
	return dst
}
