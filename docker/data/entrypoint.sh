#!/usr/bin/env sh
set -eu

cd /workspace/train
exec python -X utf8 scripts/run_data_pipeline.py "$@"
