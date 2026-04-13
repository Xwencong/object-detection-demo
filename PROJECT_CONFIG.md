# YOLOv8 GPU Linux 项目配置文档

## 项目概述

基于 Go 语言的 YOLOv8 目标检测项目，使用 ONNX Runtime GPU 加速推理。

## 系统环境

| 组件 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 22.04.5 LTS | Jammy Jellyfish |
| 架构 | x86_64 | 64位 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | Blackwell 架构 (sm_120) |
| 显存 | 8151 MiB | - |

## 核心依赖版本

### 运行时环境

| 组件 | 版本 | 安装路径 | 说明 |
|------|------|----------|------|
| Go | 1.25.0 | `/usr/local/go` | 编译和运行环境 |
| NVIDIA 驱动 | 580.126.09 | 系统级 | 支持 Blackwell 架构 |
| CUDA Toolkit | 12.8 | `/usr/local/cuda-12.8` | GPU 计算平台 |
| cuDNN | 9.20.0 | `/usr/local/cuda-12.8/lib64` | 深度学习加速库 |
| ONNX Runtime | 1.24.4 | `./third_party/` | 推理引擎（GPU 版本） |

### Go 依赖包

```go
require (
    github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
    github.com/yalue/onnxruntime_go v1.27.0
    golang.org/x/image v0.37.0
)
```

## 关键配置说明

### 1. 版本兼容性

**重要：** RTX 5060 是 2025 年发布的 Blackwell 架构 GPU (sm_120)，必须使用以下版本组合：

- **ONNX Runtime ≥ 1.24.4**：旧版本不支持 Blackwell 架构
- **onnxruntime_go v1.27.0**：对应 ONNX Runtime API 版本 20
- **CUDA ≥ 12.x**：支持新架构
- **cuDNN ≥ 9.x**：配套 CUDA 12.x

### 2. 环境变量配置

在 `~/.bashrc` 中添加：

```bash
# CUDA 环境变量
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8

# Go 环境变量
export PATH=/usr/local/go/bin:$PATH
export GOPATH=$HOME/go
export PATH=$GOPATH/bin:$PATH
export GOTOOLCHAIN=local

# Go 代理（可选，国内加速）
export GOPROXY=https://goproxy.cn,direct
```

### 3. 项目库文件

`third_party/` 目录包含运行时所需的动态库：

| 库文件 | 大小 | 用途 |
|--------|------|------|
| `onnxruntime.so` | 25M | ONNX Runtime 核心库 |
| `libonnxruntime_providers_cuda.so` | 302M | CUDA 执行提供程序 |
| `libonnxruntime_providers_shared.so` | 15K | 共享提供程序接口 |
| `libcublas.so.12` | 111M | CUDA 基础线性代数库 |
| `libcublasLt.so.12` | 717M | CUDA 轻量级 BLAS |
| `libcudart.so.12` | 712K | CUDA 运行时 |
| `libcufft.so.11` | 266M | CUDA FFT 库 |
| `libcudnn*.so.9` | ~1.8G | cuDNN 深度学习库（8个文件） |

**总大小：** 约 2.4G

## 构建和运行

### 快速开始

```bash
cd ~/project/yolov8gpu-linux
make run
```

### 常用命令

| 命令 | 说明 |
|------|------|
| `make build` | 编译项目 |
| `make run` | 编译并运行 |
| `make clean` | 清理编译产物 |
| `make check` | 检查环境配置 |
| `make help` | 显示帮助信息 |

### 手动运行

```bash
export LD_LIBRARY_PATH=./third_party:$LD_LIBRARY_PATH
./yolov8_detector_linux
```

## 性能指标

- **推理速度：** ~43ms/瓦片（1024×1024）
- **GPU 利用率：** 使用 CUDA 加速
- **支持检测：** bridge、storage tank、basketball court 等多类目标

## 项目结构

```
yolov8gpu-linux/
├── main.go                    # 主程序入口
├── global.go                  # 全局配置
├── onnx_util.go              # ONNX Runtime 工具函数
├── imagex.go                 # 图像处理
├── envi_linux.go             # Linux 环境适配
├── dll_path_linux.go         # 动态库路径配置
├── go.mod                    # Go 模块定义
├── Makefile                  # 构建脚本
├── check_env.sh              # 环境检查脚本
├── prepare_libs.sh           # 库文件准备脚本
├── yolov8m-obb.onnx          # YOLOv8 模型文件（102M）
├── data/                     # 输入数据目录
├── output_yolov8/            # 输出结果目录
└── third_party/              # 运行时依赖库（2.4G）
```

## 故障排查

### 常见问题

1. **API 版本不匹配错误**
   ```
   The requested API version [X] is not available
   ```
   - 原因：onnxruntime_go 版本与 ONNX Runtime 库版本不匹配
   - 解决：确保使用 onnxruntime_go v1.27.0 + ONNX Runtime 1.24.4

2. **GPU 不支持错误**
   ```
   CUDA error: no kernel image is available for execution
   ```
   - 原因：ONNX Runtime 版本过旧，不支持 Blackwell 架构
   - 解决：升级到 ONNX Runtime ≥ 1.24.4

3. **找不到动态库**
   ```
   error while loading shared libraries: libonnxruntime.so
   ```
   - 原因：LD_LIBRARY_PATH 未设置
   - 解决：使用 `make run` 或手动设置环境变量

### 环境检查

运行环境检查脚本：

```bash
./check_env.sh
```

会检查：
- Go 安装
- NVIDIA 驱动
- CUDA Toolkit
- CUDA 库文件
- cuDNN 库文件
- third_party 目录
- 模型文件
- 输入数据

## 更新日志

- **2026-03-26**: 初始配置完成
  - 安装 Go 1.25.0
  - 配置 CUDA 12.8 + cuDNN 9.20.0
  - 适配 RTX 5060 Blackwell 架构
  - 使用 ONNX Runtime 1.24.4
  - 项目成功运行

## 参考资源

- [ONNX Runtime 官方文档](https://onnxruntime.ai/)
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [CUDA Toolkit 文档](https://docs.nvidia.com/cuda/)
- [Go 官方文档](https://go.dev/doc/)
