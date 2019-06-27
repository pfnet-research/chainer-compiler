#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

export CHINAER_COMPILER_RUNTIME=$1

. .chainerci/run_onnx_setup_${CHINAER_COMPILER_RUNTIME}.sh

python3 utils/run_onnx_${CHINAER_COMPILER_RUNTIME}.py data/resnet50 -I 10

EOM

docker run --runtime=nvidia --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    disktnk/chainer-compiler:tvm0.5-cuda101 /bin/bash /chainer-compiler/runtest.sh ${CHINAER_COMPILER_RUNTIME}
