#!/bin/bash

set -eux

cat /proc/cpuinfo
cat /proc/meminfo
nvidia-smi

python3 -m pip install gast chainercv packaging
CHAINER_VERSION=$(python3 -c "import imp;print(imp.load_source('_version','third_party/chainer/chainer/_version.py').__version__)")
python3 -m pip install cupy-cuda101==$CHAINER_VERSION

CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS=-j8 \
    CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
    python3 -m pip -q install --no-cache-dir third_party/chainer[test]
# TODO(take-cheeze): Remove this when onnx-chainer drops 1.4.1 support
python3 -m pip install onnx==1.5.0
python3 -m pip install --no-cache-dir third_party/onnx-chainer[test]

python3 -m pip list -v
