#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

CONFIG_PATH="${POET_MINI_DATA_CONFIG:-configs/data.yaml}"
SAMPLE_ROOT="${POET_MINI_SAMPLE_ROOT:-data/sample}"
TANGSHI_DIR="${POET_MINI_TANGSHI_DIR:-${SAMPLE_ROOT}/tangshi}"

if [ ! -d "${TANGSHI_DIR}" ]; then
    echo "未找到唐诗样本目录：${TANGSHI_DIR}" >&2
    echo "请先将 chinese-poetry/全唐诗 中需要处理的 JSON 文件复制到该目录。" >&2
    exit 1
fi

if ! find "${TANGSHI_DIR}" -maxdepth 1 -type f \( -name 'poet.tang.*.json' -o -name 'poet.song.*.json' -o -name '唐诗三百首.json' -o -name '唐诗补录.json' \) | grep -q .; then
    echo "诗词样本目录中未发现可处理的诗文 JSON：${TANGSHI_DIR}" >&2
    echo "请先复制 poet.tang.*.json、poet.song.*.json、唐诗三百首.json 或 唐诗补录.json 到该目录。" >&2
    exit 1
fi

SAMPLE_FILE_COUNT="$(find "${TANGSHI_DIR}" -maxdepth 1 -type f \( -name 'poet.tang.*.json' -o -name 'poet.song.*.json' -o -name '唐诗三百首.json' -o -name '唐诗补录.json' \) | wc -l | tr -d ' ')"
echo "[build-data] 已进入 Poet-mini 容器工作区：$(pwd)"
echo "[build-data] 配置文件：${CONFIG_PATH}"
echo "[build-data] 样本目录：${TANGSHI_DIR}，可处理文件数：${SAMPLE_FILE_COUNT}"
echo "[build-data] 开始执行数据流水线..."

python -u -X utf8 scripts/run_data_pipeline.py --config "${CONFIG_PATH}" "$@"
