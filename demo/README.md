# 演示页面 (Web Demo)

一个基于标准库实现的 Web 演示页面，用于在浏览器中驱动 [根目录](../) 的 `yolov8s` 推理程序对遥感影像做目标检测。

页面本身只依赖 Go 标准库，通过 `os/exec` 调用编译好的 `yolov8s` 二进制完成实际推理，并把切片检测结果、统计信息和原始可视化图实时展示在网页上。

## 运行

在仓库根目录执行（不是在 `demo/` 目录下）：

```bash
# 1. 准备依赖库（CUDA/cuDNN/ONNX Runtime，见根目录 README）
./prepare_libs.sh

# 2. 编译推理二进制
go build -o yolov8s .
export LD_LIBRARY_PATH=$(pwd)/third_party:$LD_LIBRARY_PATH

# 3. 启动演示页面
go run ./demo
```

默认监听 `http://127.0.0.1:8092`，浏览器打开即可。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-port` | `8092` | 演示页面监听端口 |
| `-source-image` | `./fig/preprocessed.png` | 原始可视化 PNG 路径，用作页面背景底图 |
| `-binary` | `./yolov8s` | 检测程序路径 |
| `-input` | `./data/GF2_PMS1_E118.6_N37.4_20160826_L1A0001792619-MSS1-fusion.dat` | 默认检测输入 DAT 路径 |
| `-output` | `./output_yolov8s_finetune_best_class/output.jpg` | 默认检测输出路径 |
| `-classes` | `0,1,5` | 默认勾选的检测类别 ID |
| `-cpu` | `false` | 默认是否使用 CPU 推理 |
| `-labels` | `true` | 默认是否绘制标签 |

示例：

```bash
go run ./demo -port 9000 -cpu
```

## 页面功能

- **配置运行参数**：检测程序路径、DAT 输入路径、输出路径、类别勾选、CPU/标签开关。
- **上传 DAT/HDR**：也可直接在页面上传 `.dat` + `.hdr` 文件，自动写入 `demo_uploads/<时间戳>/` 并作为本次输入。
- **实时日志**：通过 Server-Sent Events 推送推理进程的标准输出。
- **切片结果浏览**：按类别分组展示每个切片的检测框图。
- **检测统计**：总目标数、各类别数量、Top 检测列表。

## 目录结构

```
demo/
├── main.go          # 演示服务（package main，仅依赖标准库）
├── web/index.html   # 前端页面（go:embed 内嵌）
└── README.md        # 本文件
```

> 注意：演示页面通过 `go:embed` 内嵌 `web/index.html`，运行时无需额外静态资源。
