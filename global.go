package main

import ort "github.com/yalue/onnxruntime_go"

var (
	UseCoreML         = false
	UseGPU            = true  // 启用 CUDA GPU 推理
	UseTensorRT       = true  // 启用 TensorRT 推理
	UseCPU            = false // 启用 CPU 推理（关闭 GPU/TensorRT）
	BatchSize         = 8     // 批量推理大小
	SaveWorkers       = 4     // 异步保存 worker 数量
	DrawLabels        = false // 性能模式默认关闭标签绘制
	VerboseDetections = false // 默认关闭逐框日志
	JPEGQuality       = 90    // 输出图像质量
	Blank             []float32
	ModelPath         = "./yolov8-obb-finetune-best-batch8.onnx"
	InputImagePath    = "./data/GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.dat"
	OutputImagePath   = "./output_yolov8s_finetune_best/output.jpg"
	Yolo8Model        ModelSession
	// EnabledClasses 为 nil 时检测全部类别；否则只保留指定类别 ID。
	EnabledClasses map[int]struct{}
	// EnvClasses 存储从 .env 读取的 CLASSES 值，供 main 解析。
	EnvClasses string
)

type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}
