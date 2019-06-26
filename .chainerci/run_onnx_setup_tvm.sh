#!/bin/bash

set -eux

. .chainerci/run_onnx_setup.sh
nvidia-smi


ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
LD_LIBRARY_PATH=/root/tvm_dist/lib:$LD_LIBRARY_PATH

CHAINER_VERSION=$(python3 -c "import imp;print(imp.load_source('_version','third_party/chainer/chainer/_version.py').__version__)")
python3 -m pip install cupy-cuda100==$CHAINER_VERSION
python3 -m pip list -v
