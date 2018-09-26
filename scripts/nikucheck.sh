#!/bin/bash
#
# Example usage:
#
# $ ./scripts/nikucheck.sh ch2o/tests/ListComp.py

set -e

mkdir -p out/tmp
PYTHONPATH=ch2o python3 "$@" out/tmp
./tools/run_onnx --test out/tmp
