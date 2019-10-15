#!/bin/bash

set -eux

. .flexci/run_onnx_setup.sh

python3 -m pip install --user 'pytest<5.0.0' 'chainercv>=0.11.0' 'onnxruntime==0.4.0'
python3 -m pip list -v
