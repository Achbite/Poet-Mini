#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Poet-mini 训练环境准备入口"
echo "当前阶段只固化容器内脚本契约，模型训练依赖将在下一阶段接入。"
"${PYTHON_BIN}" -X utf8 - <<'PY'
import sys
print(f"Python: {sys.version.split()[0]}")
PY

mkdir -p data/processed outputs/checkpoints outputs/logs outputs/samples
