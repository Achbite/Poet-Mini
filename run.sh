#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

CONFIG_PATH="configs/train.yaml"
ROLE="trainer"
RANK="0"
WORLD_SIZE="1"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"
RUN_ID="local-dev"
METRICS_PORT="9005"
DASHBOARD_PORT="9005"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
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
echo "config=${CONFIG_PATH} role=${ROLE} rank=${RANK}/${WORLD_SIZE} master=${MASTER_ADDR}:${MASTER_PORT} run_id=${RUN_ID} metrics_port=${METRICS_PORT} dashboard_port=${DASHBOARD_PORT}"
echo "当前阶段尚未接入训练主程序；下一阶段将把该入口转发到 poet_mini.training。"
