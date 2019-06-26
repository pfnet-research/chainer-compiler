#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

cat /proc/cpuinfo
cat /proc/meminfo
nvidia-smi

mkdir -p data
cd data
wget -q https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz && \
    tar -xzf resnet50.tar.gz
cd ..

python3 -m pip install --no-cache-dir third_party/onnx-chainer[test-gpu]

export PYTHONPATH=$HOME/dldt/inference-engine/bin/intel64/Release/lib/python_api/python3.6/:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/dldt/inference-engine/bin/intel64/Release/lib:$LD_LIBRARY_PATH

CHAINER_VERSION=$(python3 -c "import imp;print(imp.load_source('_version','third_party/chainer/chainer/_version.py').__version__)")
python3 -m pip install cupy-cuda100==$CHAINER_VERSION

# ngrph is skipped
for rt in onnxruntime dldt onnxruntime tvm
do
  echo Target runtime: ${rt}
  python3 utils/run_onnx_${rt}.py data/resnet50 -I 10
done

EOM

docker run --runtime=nvidia --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    disktnk/chainer-compiler:ci-base-7c293fc /bin/bash /chainer-compiler/runtest.sh
