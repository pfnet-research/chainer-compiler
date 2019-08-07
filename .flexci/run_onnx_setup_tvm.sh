#!/bin/bash

set -eux

. .chainerci/run_onnx_setup.sh
nvidia-smi

export LD_LIBRARY_PATH=/root/tvm_dist/lib:$LD_LIBRARY_PATH

CHAINER_VERSION=$(python3 -c "import imp;print(imp.load_source('_version','third_party/chainer/chainer/_version.py').__version__)")
python3 -m pip install cupy-cuda101==$CHAINER_VERSION onnx==1.5.0
python3 -m pip list -v
