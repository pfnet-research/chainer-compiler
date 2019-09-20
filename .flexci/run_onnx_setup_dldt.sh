#!/bin/bash

set -eux

. .flexci/run_onnx_setup.sh

export LD_LIBRARY_PATH=/root/dldt_dist/lib:/root/dldt/inference-engine/bin/intel64/Release/lib/
export PYTHONPATH=/root/dldt/model-optimizer:/root/dldt/inference-engine/bin/intel64/Release/lib/python_api/python3.6/

python3 -m pip list -v
