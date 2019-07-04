#!/bin/bash

set -eu

SCRIPT_DIR=$(dirname "$0")
QUANTIZER_LIB="$SCRIPT_DIR/../build/quantize.py"

if [ ! -f "$QUANTIZER_LIB" ] ; then
    curl -L 'https://raw.githubusercontent.com/microsoft/onnxruntime/master/onnxruntime/python/tools/quantization/quantize.py' -o "$QUANTIZER_LIB"
fi

export PYTHONPATH="$SCRIPT_DIR/../build"
python "$SCRIPT_DIR/../scripts/quantize_model.py" $@
