#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname "$0")

export PYTHONPATH="$SCRIPT_DIR/../third_party/onnxruntime/onnxruntime/python/tools/quantization"
python "$SCRIPT_DIR/../scripts/quantize_model.py" $1 $2
