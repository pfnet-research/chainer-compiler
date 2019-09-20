#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

export CHINAER_COMPILER_RUNTIME=$1

. .flexci/run_onnx_setup_${CHINAER_COMPILER_RUNTIME}.sh

python3 utils/run_onnx_${CHINAER_COMPILER_RUNTIME}.py data/shufflenet -I 10

EOM

. .flexci/common.sh

docker run ${CHINAER_COMPILER_DOCKER_RUNTIME_ARG} --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    ${CI_IMAGE} /bin/bash /chainer-compiler/runtest.sh ${CHINAER_COMPILER_RUNTIME}
