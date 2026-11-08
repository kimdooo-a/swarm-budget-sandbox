#!/usr/bin/env sh
# Rerun one episode with a fixed seed: ./rerun.sh --seed 3 [--condition pressure] [--n 8]
# Rebuild every table and figure from the saved traces: ./rerun.sh --seed 0 --analysis-only
cd "$(dirname "$0")" && exec python rerun.py "$@"
