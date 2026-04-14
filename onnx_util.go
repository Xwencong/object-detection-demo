package main

import (
	"fmt"
	"path/filepath"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

func InitOrtEnv() error {
	ort.SetSharedLibraryPath(getSharedLibPath())
	return ort.InitializeEnvironment()
}

func InitYolo8Session(input []float32) (ModelSession, error) {
	// 使用 BatchSize 创建 tensor
	inputShape := ort.NewShape(int64(BatchSize), 3, 640, 640)
	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return ModelSession{}, err
	}

	outputShape := ort.NewShape(int64(BatchSize), 13, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return ModelSession{}, err
	}

	options, e := ort.NewSessionOptions()
	if e != nil {
		return ModelSession{}, e
	}
	defer options.Destroy()

	if UseTensorRT {
		// TensorRT EP（最快，首次运行会构建引擎缓存）
		trtOptions, e := ort.NewTensorRTProviderOptions()
		if e != nil {
			return ModelSession{}, fmt.Errorf("创建 TensorRT 选项失败: %w", e)
		}
		e = trtOptions.Update(map[string]string{
			"device_id":                 "0",
			"trt_fp16_enable":           "1", // 启用 FP16 加速
			"trt_engine_cache_enable":   "1", // 启用引擎缓存
			"trt_engine_cache_path":     "./trt_cache",
			"trt_max_workspace_size":    "2147483648", // 2GB
		})
		if e != nil {
			trtOptions.Destroy()
			return ModelSession{}, fmt.Errorf("更新 TensorRT 选项失败: %w", e)
		}
		e = options.AppendExecutionProviderTensorRT(trtOptions)
		trtOptions.Destroy()
		if e != nil {
			return ModelSession{}, fmt.Errorf("添加 TensorRT EP 失败: %w", e)
		}
	} else if UseGPU {
		cudaOptions, e := ort.NewCUDAProviderOptions()
		if e != nil {
			return ModelSession{}, fmt.Errorf("创建 CUDA 选项失败: %w", e)
		}
		e = cudaOptions.Update(map[string]string{
			"device_id": "0",
		})
		if e != nil {
			cudaOptions.Destroy()
			return ModelSession{}, fmt.Errorf("更新 CUDA 选项失败: %w", e)
		}
		e = options.AppendExecutionProviderCUDA(cudaOptions)
		cudaOptions.Destroy()
		if e != nil {
			return ModelSession{}, fmt.Errorf("添加 CUDA EP 失败: %w", e)
		}
	} else if UseCoreML { // If CoreML is enabled, append the CoreML execution provider
		e = options.AppendExecutionProviderCoreML(0)
		if e != nil {
			return ModelSession{}, fmt.Errorf("添加 CoreML EP 失败: %w", e)
		}
	}

	session, err := ort.NewAdvancedSession(ModelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, options)

	if err != nil {
		return ModelSession{}, err
	}

	modelSes := ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}

	return modelSes, err
}

func getSharedLibPath() string {
	rel := ""
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		rel = "./third_party/onnxruntime.dll"
	} else if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		rel = "./third_party/onnxruntime_arm64.dylib"
	} else if runtime.GOOS == "linux" && runtime.GOARCH == "arm64" {
		rel = "./third_party/onnxruntime_arm64.so"
	} else if runtime.GOOS == "linux" {
		rel = "./third_party/onnxruntime.so"
	} else {
		panic("Unable to find a version of the onnxruntime library supporting this system.")
	}
	abs, err := filepath.Abs(rel)
	if err != nil {
		return rel
	}
	return abs
}

// 微调模型类别（labels-v2）
var yolo_classes = []string{
	"oil", "bridge", "basketball court", "ground track field",
	"roundabout", "harbor", "other court", "soccer ball field",
}

// 类别中文名称
var yolo_classes_zh = map[string]string{
	"oil":                "油罐",
	"bridge":             "桥梁",
	"basketball court":   "篮球场",
	"ground track field": "田径场",
	"roundabout":         "环形交叉口",
	"harbor":             "码头",
	"other court":        "其他球场",
	"soccer ball field":  "足球场",
}

func runInference(modelSes ModelSession, input []float32) ([]float32, error) {
	inTensor := modelSes.Input.GetData()
	copy(inTensor, input)
	err := modelSes.Session.Run()
	if err != nil {
		return nil, err
	}
	return modelSes.Output.GetData(), nil
}

// runBatchInference 批量推理，inputs 是多个瓦片输入。
// 返回 output tensor 中对应每个瓦片的切片视图；调用方必须在下一次 Run 前消费完。
func runBatchInference(modelSes ModelSession, inputs [][]float32) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	batchSize := len(inputs)
	inTensor := modelSes.Input.GetData()
	singleSize := 3 * 640 * 640

	for i, input := range inputs {
		copy(inTensor[i*singleSize:(i+1)*singleSize], input)
	}

	if batchSize < BatchSize {
		if len(Blank) != singleSize {
			Blank = make([]float32, singleSize)
		}
		for i := batchSize; i < BatchSize; i++ {
			copy(inTensor[i*singleSize:(i+1)*singleSize], Blank)
		}
	}

	err := modelSes.Session.Run()
	if err != nil {
		return nil, err
	}

	outTensor := modelSes.Output.GetData()
	outputs := make([][]float32, batchSize)
	outputSize := 13 * 8400
	for i := 0; i < batchSize; i++ {
		start := i * outputSize
		outputs[i] = outTensor[start : start+outputSize]
	}

	return outputs, nil
}
