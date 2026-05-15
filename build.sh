#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "${PYTHON_BIN}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "未找到 python 或 python3，请先进入容器或安装 Python。" >&2
        exit 1
    fi
fi

echo "Poet-mini 训练环境准备入口"
echo "Python 命令：${PYTHON_BIN}"
"${PYTHON_BIN}" -X utf8 - <<'PY'
import sys
print(f"Python: {sys.version.split()[0]}")
PY

echo "训练依赖已改为 Docker 镜像构建阶段安装，当前脚本只做环境检查。"

"${PYTHON_BIN}" -X utf8 - <<'PY'
import sys
missing = []
for name in ("yaml", "numpy", "torch"):
    try:
        __import__(name)
    except ImportError:
        missing.append(name)
if missing:
    print("缺少训练依赖：" + ", ".join(missing))
    print("请退出容器后执行：bash makeshell -clean -gpu")
    sys.exit(1)
import torch
print(f"Torch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available() and torch.version.cuda:
    print("提示：如果容器已使用 -gpu 创建但 CUDA 不可用，请确认 PyTorch CUDA 版本不高于宿主 NVIDIA Driver 支持的 CUDA 版本。")
PY

mkdir -p data/processed outputs/checkpoints outputs/logs outputs/samples
