#!/bin/bash

# 环境检查脚本

echo "=== YOLOv8 GPU Linux 环境检查 ==="
echo ""

# 检查 Go 版本
echo "1. 检查 Go 安装..."
if command -v go &> /dev/null; then
    GO_VERSION=$(go version)
    echo "   ✓ $GO_VERSION"
else
    echo "   ✗ Go 未安装"
    exit 1
fi

# 检查 NVIDIA 驱动
echo ""
echo "2. 检查 NVIDIA 驱动..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ NVIDIA 驱动已安装"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
        echo "     GPU: $line"
    done
else
    echo "   ✗ NVIDIA 驱动未安装或不可用"
fi

# 检查 CUDA
echo ""
echo "3. 检查 CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "   ✓ CUDA 版本: $CUDA_VERSION"
else
    echo "   ✗ CUDA Toolkit 未安装"
fi

# 检查 CUDA 库路径
echo ""
echo "4. 检查 CUDA 库..."
CUDA_LIBS=(
    "/usr/local/cuda/lib64/libcublas.so.12"
    "/usr/local/cuda/lib64/libcublasLt.so.12"
    "/usr/local/cuda/lib64/libcudart.so.12"
    "/usr/local/cuda/lib64/libcufft.so.11"
)

for lib in "${CUDA_LIBS[@]}"; do
    if [ -f "$lib" ]; then
        echo "   ✓ $(basename $lib)"
    else
        echo "   ✗ $(basename $lib) 未找到"
    fi
done

# 检查 cuDNN
echo ""
echo "5. 检查 cuDNN 库..."
CUDNN_LIBS=(
    "/usr/local/cuda/lib64/libcudnn.so.9"
    "/usr/local/cuda/lib64/libcudnn_ops.so.9"
    "/usr/local/cuda/lib64/libcudnn_cnn.so.9"
)

for lib in "${CUDNN_LIBS[@]}"; do
    if [ -f "$lib" ]; then
        echo "   ✓ $(basename $lib)"
    else
        echo "   ✗ $(basename $lib) 未找到"
    fi
done

# 检查 third_party 目录
echo ""
echo "6. 检查 third_party 目录..."
if [ -d "./third_party" ]; then
    echo "   ✓ third_party 目录存在"

    # 检查关键库文件
    REQUIRED_LIBS=(
        "onnxruntime.so"
        "onnxruntime_providers_cuda.so"
        "onnxruntime_providers_shared.so"
    )

    for lib in "${REQUIRED_LIBS[@]}"; do
        if [ -f "./third_party/$lib" ]; then
            echo "   ✓ $lib"
        else
            echo "   ✗ $lib 缺失"
        fi
    done
else
    echo "   ✗ third_party 目录不存在"
fi

# 检查模型文件
echo ""
echo "7. 检查模型文件..."
if [ -f "./yolov8m-obb.onnx" ]; then
    SIZE=$(du -h "./yolov8m-obb.onnx" | cut -f1)
    echo "   ✓ yolov8m-obb.onnx ($SIZE)"
else
    echo "   ✗ yolov8m-obb.onnx 未找到"
fi

# 检查输入数据
echo ""
echo "8. 检查输入数据..."
if [ -d "./data" ]; then
    DAT_COUNT=$(find ./data -name "*.dat" | wc -l)
    echo "   ✓ data 目录存在 ($DAT_COUNT 个 .dat 文件)"
else
    echo "   ✗ data 目录不存在"
fi

echo ""
echo "=== 检查完成 ==="
echo ""
echo "建议:"
echo "- 如果缺少 CUDA 库，请从 /usr/local/cuda/lib64/ 复制到 third_party/"
echo "- 如果缺少 ONNX Runtime 库，请从官网下载 GPU 版本"
echo "- 运行前设置: export LD_LIBRARY_PATH=\$(pwd)/third_party:\$LD_LIBRARY_PATH"
