#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

. .chainerci/before_install.sh

CUDNN_ROOT_DIR=/usr/local/cuda/lib MAKEFLAGS=-j8 python3 -m pip install -e . -vvv

for gen in `ls scripts | grep -E "^gen_.*\.py$"`
do
  python3 scripts/${gen}
done
for dir in model node node/ndarray node/Functions node/Links syntax
do
  PYTHONPATH=. python3 scripts/elichika_tests.py --generate ${dir}
done

for dir in model node syntax
do
  for op in `ls testcases/ch2o_tests/$dir | sed 's/\.[^\.]*$//'`
  do
    PYTHONPATH=. python3 testcases/ch2o_tests/${dir}/${op}.py out/ch2o_${dir}_${op} --quiet && true
  done
done

# scripts/runtests.py needs run_onnx binary, skip the test

PYTHONPATH=. python3 -m pytest -sv tests

EOM

docker run --runtime=nvidia --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    disktnk/chainer-compiler:ci-cuda100-ngraph0.19.0-tvm0.5 /bin/bash /chainer-compiler/runtest.sh
