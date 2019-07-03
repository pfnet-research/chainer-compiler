#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

export CHINAER_COMPILER_RUNTIME=$1

. .chainerci/run_onnx_setup_${CHINAER_COMPILER_RUNTIME}.sh

python3 utils/run_onnx_${CHINAER_COMPILER_RUNTIME}.py data/resnet50 -I 10

EOM

docker run ${CHINAER_COMPILER_DOCKER_RUNTIME_ARG} --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    disktnk/chainer-compiler:ci-base-22b692b /bin/bash /chainer-compiler/runtest.sh ${CHINAER_COMPILER_RUNTIME}
