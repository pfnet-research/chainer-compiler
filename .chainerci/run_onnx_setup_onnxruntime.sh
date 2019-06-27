#!/bin/bash

set -eux

. .chainerci/run_onnx_setup.sh

python3 -m pip install --user -e third_party/onnx-chainer[test-cpu]
python3 -m pip list -v
