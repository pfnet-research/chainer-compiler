#!/bin/bash

set -eu

SCRIPT_DIR=$(dirname "$0")

if [ ! -f "$SCRIPT_DIR/../scripts/quantize.py" ] ; then
    curl -L 'https://raw.githubusercontent.com/microsoft/onnxruntime/master/onnxruntime/python/tools/quantization/quantize.py' -o "$SCRIPT_DIR/../build/quantize.py"
fi

export PYTHONPATH="$SCRIPT_DIR/../build"
python "$SCRIPT_DIR/../scripts/quantize_model.py" $@
