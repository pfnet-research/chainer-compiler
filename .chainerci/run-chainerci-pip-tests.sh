#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

. .chainerci/before_install.sh

CUDNN_ROOT_DIR=/usr/local/cuda/lib MAKEFLAGS=-j8 python3 -m pip install -e . -vvv

for gen in `ls scripts | grep -E "^gen_.*\.py$"`
do
  python3 scripts/$gen
done
for dir in model node node/ndarray node/Functions node/Links syntax
do
  PYTHONPATH=. python3 scripts/elichika_tests.py --generate $dir
done

PYTHONPATH=. python3 scripts/runtests.py "(^(test_))|(^(backprop_))|(^(extra_))|(^(elichika_)).*$"  -g --fuse
PYTHONPATH=. python3 -m pytest -sv tests

EOM

docker run --runtime=nvidia --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    disktnk/chainer-compiler:ci-cuda100-ngraph0.19.0-tvm0.5 /bin/bash /chainer-compiler/runtest.sh
