.PHONY: all build clean check prepare run help

# 变量定义
BINARY_NAME=yolov8_detector_linux
THIRD_PARTY_DIR=./third_party
GO=go
GOFLAGS=-v

# 默认目标
all: check build

# 构建项目
build:
	@echo "正在构建 $(BINARY_NAME)..."
	@export LD_LIBRARY_PATH=$(THIRD_PARTY_DIR):$$LD_LIBRARY_PATH && \
	$(GO) build $(GOFLAGS) -o $(BINARY_NAME) .
	@echo "构建完成: $(BINARY_NAME)"

# 检查环境
check:
	@echo "检查环境..."
	@./check_env.sh || true

# 准备库文件
prepare:
	@echo "准备库文件..."
	@./prepare_libs.sh

# 运行程序
run: build
	@echo "运行 $(BINARY_NAME)..."
	@export LD_LIBRARY_PATH=$(THIRD_PARTY_DIR):$$LD_LIBRARY_PATH && \
	./$(BINARY_NAME)

# 清理构建产物
clean:
	@echo "清理..."
	@rm -f $(BINARY_NAME)
	@rm -rf output_yolov8/*
	@echo "清理完成"

# 深度清理（包括库文件）
clean-all: clean
	@echo "深度清理..."
	@rm -rf $(THIRD_PARTY_DIR)/*.so
	@echo "深度清理完成"

# 安装依赖
deps:
	@echo "安装 Go 依赖..."
	@$(GO) mod download
	@$(GO) mod tidy

# 测试构建
test-build:
	@echo "测试构建（不链接）..."
	@export LD_LIBRARY_PATH=$(THIRD_PARTY_DIR):$$LD_LIBRARY_PATH && \
	$(GO) build -o /dev/null .

# 显示帮助
help:
	@echo "YOLOv8 GPU Linux Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  make all        - 检查环境并构建（默认）"
	@echo "  make build      - 构建项目"
	@echo "  make check      - 检查环境配置"
	@echo "  make prepare    - 准备库文件"
	@echo "  make run        - 构建并运行"
	@echo "  make clean      - 清理构建产物"
	@echo "  make clean-all  - 深度清理（包括库文件）"
	@echo "  make deps       - 安装 Go 依赖"
	@echo "  make test-build - 测试构建"
	@echo "  make help       - 显示此帮助信息"
	@echo ""
	@echo "环境变量:"
	@echo "  LD_LIBRARY_PATH - 自动设置为 $(THIRD_PARTY_DIR)"
	@echo ""
	@echo "快速开始:"
	@echo "  1. make prepare  # 准备库文件"
	@echo "  2. make check    # 检查环境"
	@echo "  3. make build    # 构建项目"
	@echo "  4. make run      # 运行程序"
