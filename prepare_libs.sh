#!/bin/bash

# 准备 Linux 库文件脚本
#
# 所需依赖版本（版本必须严格匹配，否则会导致 ORT 初始化失败）：
#   CUDA              12.8
#   cuDNN             9.2.0
#   TensorRT          10.16.0
#   ONNX Runtime GPU  1.24.4
#
# ONNX Runtime 下载地址：
#   https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-gpu-1.24.4.tgz
#   解压后执行：
#     cp onnxruntime-linux-x64-gpu-1.24.4/lib/*.so third_party/
#     mv third_party/libonnxruntime.so.1.24.4 third_party/onnxruntime.so
#     ln -s libonnxruntime_providers_cuda.so third_party/onnxruntime_providers_cuda.so
#     ln -s libonnxruntime_providers_shared.so third_party/onnxruntime_providers_shared.so

echo "=== 准备 Linux 运行库 ==="
echo ""

THIRD_PARTY_DIR="./third_party"

# 创建 third_party 目录（如果不存在）
mkdir -p "$THIRD_PARTY_DIR"

echo "1. 检查 CUDA 库..."
CUDA_LIB_DIR="/usr/local/cuda/lib64"

if [ -d "$CUDA_LIB_DIR" ]; then
    echo "   找到 CUDA 库目录: $CUDA_LIB_DIR"

    # 复制 CUDA 库
    CUDA_LIBS=(
        "libcublas.so.12"
        "libcublasLt.so.12"
        "libcudart.so.12"
        "libcufft.so.11"
    )

    for lib in "${CUDA_LIBS[@]}"; do
        if [ -f "$CUDA_LIB_DIR/$lib" ]; then
            if [ ! -f "$THIRD_PARTY_DIR/$lib" ]; then
                echo "   复制 $lib..."
                cp "$CUDA_LIB_DIR/$lib" "$THIRD_PARTY_DIR/"
            else
                echo "   ✓ $lib 已存在"
            fi
        else
            echo "   ✗ $lib 未找到"
        fi
    done
else
    echo "   ✗ CUDA 库目录不存在: $CUDA_LIB_DIR"
fi

echo ""
echo "2. 检查 cuDNN 库..."

# cuDNN 可能在不同位置
CUDNN_DIRS=(
    "/usr/local/cuda/lib64"
    "/usr/lib/x86_64-linux-gnu"
)

CUDNN_LIBS=(
    "libcudnn.so.9"
    "libcudnn_adv.so.9"
    "libcudnn_cnn.so.9"
    "libcudnn_engines_precompiled.so.9"
    "libcudnn_engines_runtime_compiled.so.9"
    "libcudnn_graph.so.9"
    "libcudnn_heuristic.so.9"
    "libcudnn_ops.so.9"
)

for lib in "${CUDNN_LIBS[@]}"; do
    found=false
    for dir in "${CUDNN_DIRS[@]}"; do
        if [ -f "$dir/$lib" ]; then
            if [ ! -f "$THIRD_PARTY_DIR/$lib" ]; then
                echo "   复制 $lib..."
                cp "$dir/$lib" "$THIRD_PARTY_DIR/"
            else
                echo "   ✓ $lib 已存在"
            fi
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "   ✗ $lib 未找到"
    fi
done

echo ""
echo "3. 检查 ONNX Runtime 库..."

ORT_LIBS=(
    "onnxruntime.so"
    "libonnxruntime_providers_cuda.so"
    "libonnxruntime_providers_shared.so"
    "libonnxruntime_providers_tensorrt.so"
)

all_found=true
for lib in "${ORT_LIBS[@]}"; do
    if [ -f "$THIRD_PARTY_DIR/$lib" ]; then
        echo "   ✓ $lib 已存在"
    else
        echo "   ✗ $lib 未找到"
        all_found=false
    fi
done

if [ "$all_found" = false ]; then
    echo ""
    echo "   请手动下载 ONNX Runtime GPU 1.19.2："
    echo "   1. 访问：https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2"
    echo "   2. 下载：onnxruntime-linux-x64-gpu-1.19.2.tgz"
    echo "   3. 解压后将 lib/ 目录下的 .so 文件复制到 $THIRD_PARTY_DIR/"
    echo "   4. 将 libonnxruntime.so.1.19.2 重命名为 onnxruntime.so"
    echo "   5. 创建软链接："
    echo "      ln -s libonnxruntime_providers_cuda.so $THIRD_PARTY_DIR/onnxruntime_providers_cuda.so"
    echo "      ln -s libonnxruntime_providers_shared.so $THIRD_PARTY_DIR/onnxruntime_providers_shared.so"
fi

echo ""
echo "=== 库文件列表 ==="
ls -lh "$THIRD_PARTY_DIR"/*.so 2>/dev/null || echo "没有找到 .so 文件"

echo ""
echo "=== 完成 ==="
echo ""
echo "下一步:"
echo "1. 确保所有必需的 .so 文件都在 third_party/ 目录"
echo "2. 运行: export LD_LIBRARY_PATH=\$(pwd)/third_party:\$LD_LIBRARY_PATH"
echo "3. 构建: go build -o yolov8_detector_linux ."
echo "4. 运行: ./yolov8_detector_linux"
