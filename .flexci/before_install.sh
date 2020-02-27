#!/bin/bash

set -eux

cat /proc/cpuinfo
cat /proc/meminfo
nvidia-smi

CHAINER_VERSION=$(python3 -c "import imp;print(imp.load_source('_version','third_party/chainer/chainer/_version.py').__version__)")
python3 -m pip install cupy-cuda101==$CHAINER_VERSION

if [[ -d dist-chainer ]]; then
    echo "Use cached chainer wheel"
else
    pushd third_party/chainer
    python3 -m pip install wheel
    CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS=-j8 \
        CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
        python3 setup.py bdist_wheel -vvv --bdist-dir ~/temp/bdistwheel
    popd
    mkdir dist-chainer
    cp -p third_party/chainer/dist/*.whl dist-chainer
fi
python3 -m pip install dist-chainer/*.whl

python3 -m pip install gast chainercv torch==1.4.0
# TODO(take-cheeze): Remove this when onnx-chainer drops 1.4.1 support
python3 -m pip install onnx==1.5.0 'pytest<5.0.0' 'chainercv>=0.11.0' 'packaging>=19.0'

python3 -m pip list -v
