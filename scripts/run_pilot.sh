#!/usr/bin/env bash
set -euo pipefail

python -m src.inference.run_experiment --config configs/inference.json "$@"
