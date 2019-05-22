#!/bin/bash
#
# How to update third_party module:
#
# $ ./scripts/bump_submodule.sh third_party/chainer
# $ (cd build && ninja)  # or make
# $ # Fix build, if necessary.
# $ ./scripts/runtests.py
# $ git push origin bump-chainer
#
# Then send the PR to check if CI is happy.
#
# The following is necessary only for Chainer. You can skip these
# steps if CI says it is OK.
#
# $ pip3 install --user third_party/chainer
# $ pytest python
#
# More tips:
#
# - When you update Chainer, you should review the result of
#   `git log chainerx_cc/chainerx/routines` to see updated/added ops.
# - When you update ONNX, it is also good to see changes for the
#   standard ops.
#

set -eux

submodule="$1"
name=$(basename "${submodule}")

git branch -D bump-${name} || echo 'Branch delete error ignored'
git checkout -b bump-${name}

pushd "${submodule}"
git fetch
git checkout origin/master
commit=$(git log | head -1 | awk '{print $2}')
popd

git commit -a -m "Bump up ${name} version

to ${commit}
"
