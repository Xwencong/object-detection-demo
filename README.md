# 基于 Go 的 YOLOv8 目标检测推理

本项目使用 [Go](https://go.dev) 实现了基于 [YOLOv8 目标检测神经网络](https://ultralytics.com/yolov8) 的遥感图像目标检测推理服务。

## 依赖库说明

`third_party/` 目录包含 CUDA/cuDNN/ONNX Runtime 等大型共享库（总计约 2GB），由于 GitHub 文件大小限制，**这些文件未包含在本仓库中**。

运行以下脚本在本地配置依赖库：

```bash
./prepare_libs.sh
```

该脚本会从本地 CUDA 安装目录将所需 `.so` 文件复制到 `third_party/`。ONNX Runtime GPU 库请按脚本输出的提示手动下载。

所需库文件：
- CUDA 12：`libcublas.so.12`、`libcublasLt.so.12`、`libcudart.so.12`、`libcufft.so.11`
- cuDNN 9：`libcudnn*.so.9`
- ONNX Runtime GPU：`onnxruntime.so`、`libonnxruntime_providers_cuda.so` 等

## 安装

* 克隆本仓库：`git clone https://github.com/PCL-Satellite-Cloud-Native/Object-Detection.git`
* 进入仓库根目录
* 配置第三方依赖库：`./prepare_libs.sh`
* 安装 Go 依赖：`go get`

## 运行

1. 编译二进制文件：

```bash
go build -o yolov8gpu-linux .
```

2. 执行目标检测：

```bash
export LD_LIBRARY_PATH=$(pwd)/third_party:$LD_LIBRARY_PATH
./yolov8gpu-linux -classes "0,1,5" -labels -input ./data/GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.dat -output ./output_yolov8s_finetune_best_class/output.jpg
```

参数说明：
- `-classes` — 指定要检测的目标类别 ID，逗号分隔（如 `"0,1,5"`）；不指定则检测所有类别
- `-labels` — 在输出图像上绘制类别标签（可选）
- `-input` — 输入图像或遥感数据文件路径
- `-output` — 检测结果输出图像路径
