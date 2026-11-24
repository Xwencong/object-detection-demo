# 基于 Go 的 YOLOv8 目标检测推理

本项目使用 [Go](https://go.dev) 实现了基于 [YOLOv8 目标检测神经网络](https://ultralytics.com/yolov8) 的遥感图像目标检测推理服务，支持 GPU（TensorRT 加速）和 CPU 两种推理模式。

仓库附带一个 Web 演示页面（`demo/`），可在浏览器中驱动推理程序并实时查看切片检测结果，详见 [demo/README.md](demo/README.md)。启动方式：

```bash
go build -o yolov8s .
export LD_LIBRARY_PATH=$(pwd)/third_party:$LD_LIBRARY_PATH
go run ./demo
```

默认打开 `http://127.0.0.1:8092`。

## 测试数据

`data/` 目录包含测试用遥感图像（总计约 3.1GB），由于文件过大**未包含在本仓库中**。

| 文件 | 大小 | 说明 |
|------|------|------|
| `GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.dat` | 3.1G | 高分二号遥感影像数据体（ENVI BSQ 格式） |
| `GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.hdr` | 831B | ENVI 头文件，记录影像元数据 |

影像基本信息：
- 卫星：高分二号（GF-2）PMS1 传感器
- 成像时间：2016-08-26
- 中心坐标：东经 118.6°，北纬 37.4°
- 尺寸：23365 × 23365 像素，3 波段
- 空间分辨率：约 0.846m
- 坐标系：WGS-84 / UTM Zone 50N

---

## GPU 模式（TensorRT 加速，推荐）

### 依赖版本

| 依赖 | 版本 |
|------|------|
| CUDA | 12.8 |
| cuDNN | 9.2.0 |
| TensorRT | 10.16.0 |
| ONNX Runtime GPU | 1.24.4 |

> 注意：以上版本必须严格匹配，版本不一致会导致 `ORT 初始化失败` 错误。

### 所需库文件清单

**CUDA 12（共 4 个文件，从 `/usr/local/cuda/lib64/` 复制）：**

| 文件 | 大小 |
|------|------|
| `libcublas.so.12` | 111M |
| `libcublasLt.so.12` | 717M |
| `libcudart.so.12` | 712K |
| `libcufft.so.11` | 266M |

**cuDNN 9（共 8 个文件，从 `/usr/local/cuda/lib64/` 或 `/usr/lib/x86_64-linux-gnu/` 复制）：**

| 文件 | 大小 |
|------|------|
| `libcudnn.so.9` | 127K |
| `libcudnn_adv.so.9` | 259M |
| `libcudnn_cnn.so.9` | 3.6M |
| `libcudnn_engines_precompiled.so.9` | 492M |
| `libcudnn_engines_runtime_compiled.so.9` | 29M |
| `libcudnn_graph.so.9` | 3.6M |
| `libcudnn_heuristic.so.9` | 57M |
| `libcudnn_ops.so.9` | 102M |

**ONNX Runtime GPU 1.24.4（共 6 个文件）：**

| 文件 | 大小 | 说明 |
|------|------|------|
| `onnxruntime.so` | 25M | 主库（`libonnxruntime.so` 软链接或重命名） |
| `libonnxruntime_providers_cuda.so` | 302M | CUDA 推理后端 |
| `libonnxruntime_providers_shared.so` | 15K | 共享库 |
| `libonnxruntime_providers_tensorrt.so` | 835K | TensorRT 推理后端 |
| `onnxruntime_providers_cuda.so` | — | 软链接 → `libonnxruntime_providers_cuda.so` |
| `onnxruntime_providers_shared.so` | — | 软链接 → `libonnxruntime_providers_shared.so` |

### 配置依赖库

运行脚本自动复制 CUDA/cuDNN 库，并自动下载 ONNX Runtime：

```bash
./prepare_libs.sh
```

或手动下载 ONNX Runtime GPU 1.24.4：

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-gpu-1.24.4.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.24.4.tgz
cp onnxruntime-linux-x64-gpu-1.24.4/lib/*.so third_party/
ln -s libonnxruntime.so third_party/onnxruntime.so          # 若无 onnxruntime.so
ln -s libonnxruntime_providers_cuda.so third_party/onnxruntime_providers_cuda.so
ln -s libonnxruntime_providers_shared.so third_party/onnxruntime_providers_shared.so
```

### 安装与运行

```bash
git clone https://github.com/Xwencong/object-detection-demo.git
cd object-detection-demo
./prepare_libs.sh
go get
go build -o yolov8s .
export LD_LIBRARY_PATH=$(pwd)/third_party:$LD_LIBRARY_PATH
./yolov8s -classes "0,1,5" -labels \
  -input ./data/GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.dat \
  -output ./output_yolov8s_finetune_best_class/output.jpg
```

---

## CPU 模式（无需 GPU 环境）

### 依赖版本

| 依赖 | 版本 |
|------|------|
| ONNX Runtime CPU | 1.24.4 |

### 所需库文件清单

仅需 ONNX Runtime 主库（共 3 个文件，放入 `third_party/`）：

| 文件 | 说明 |
|------|------|
| `onnxruntime.so` | 主库（`libonnxruntime.so` 软链接或重命名） |
| `libonnxruntime_providers_shared.so` | 共享库 |
| `onnxruntime_providers_shared.so` | 软链接 → `libonnxruntime_providers_shared.so` |

### 配置依赖库

下载 ONNX Runtime CPU 1.24.4：

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz
tar -xzf onnxruntime-linux-x64-1.24.4.tgz
mkdir -p third_party
cp onnxruntime-linux-x64-1.24.4/lib/*.so third_party/
ln -s libonnxruntime.so third_party/onnxruntime.so          # 若无 onnxruntime.so
ln -s libonnxruntime_providers_shared.so third_party/onnxruntime_providers_shared.so
```

### 安装与运行

```bash
git clone https://github.com/Xwencong/object-detection-demo.git
cd object-detection-demo
# 配置依赖库（见上方）
go get
go build -o yolov8s .
export LD_LIBRARY_PATH=$(pwd)/third_party:$LD_LIBRARY_PATH
./yolov8s -cpu -classes "0,1,5" -labels \
  -input ./data/GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.dat \
  -output ./output_yolov8s_finetune_best_class/output.jpg
```

---

## 参数说明

| 参数 | 说明 |
|------|------|
| `-cpu` | 使用 CPU 推理，禁用 GPU/TensorRT（默认为 GPU 模式） |
| `-classes` | 指定检测类别 ID，逗号分隔（如 `"0,1,5"`）；不指定则检测全部类别 |
| `-labels` | 在输出图像上绘制类别标签（可选） |
| `-input` | 输入图像或遥感数据文件路径 |
| `-output` | 检测结果输出图像路径 |
| `-verbose` | 输出逐框检测日志（可选） |
