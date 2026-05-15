#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

TRAIN_CONFIG_PATH="configs/train.yaml"
MODEL_CONFIG_PATH="configs/model.yaml"
ROLE="trainer"
RANK="0"
WORLD_SIZE="1"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"
RUN_ID="local-dev"
METRICS_PORT="9005"
DASHBOARD_PORT="9005"
RESUME_PATH=""
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

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            TRAIN_CONFIG_PATH="$2"
            shift 2
            ;;
        --model-config)
            MODEL_CONFIG_PATH="$2"
            shift 2
            ;;
        --role)
            ROLE="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --metrics-port)
            METRICS_PORT="$2"
            shift 2
            ;;
        --dashboard-port)
            DASHBOARD_PORT="$2"
            shift 2
            ;;
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        *)
            echo "未知参数：$1" >&2
            exit 2
            ;;
    esac
done

export POET_MINI_ROLE="${ROLE}"
export RANK="${RANK}"
export WORLD_SIZE="${WORLD_SIZE}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export POET_MINI_RUN_ID="${RUN_ID}"
export POET_MINI_METRICS_PORT="${METRICS_PORT}"
export POET_MINI_DASHBOARD_PORT="${DASHBOARD_PORT}"

echo "Poet-mini 训练启动入口"
echo "train_config=${TRAIN_CONFIG_PATH} model_config=${MODEL_CONFIG_PATH} role=${ROLE} rank=${RANK}/${WORLD_SIZE} master=${MASTER_ADDR}:${MASTER_PORT} run_id=${RUN_ID}"

if [ "${ROLE}" != "trainer" ]; then
    echo "当前最小训练闭环仅支持 role=trainer" >&2
    exit 2
fi

if [ -n "${RESUME_PATH}" ]; then
    exec "${PYTHON_BIN}" -m poet_mini.training.train --model-config "${MODEL_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}" --run-id "${RUN_ID}" --resume "${RESUME_PATH}"
fi

exec "${PYTHON_BIN}" -m poet_mini.training.train --model-config "${MODEL_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}" --run-id "${RUN_ID}"
