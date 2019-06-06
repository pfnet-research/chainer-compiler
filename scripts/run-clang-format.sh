#!/usr/bin/env bash

set -eu

cd $(git rev-parse --show-toplevel)

git ls-files | grep -e '\.cc$\|\.h$' | xargs -P4 clang-format -i
git diff --exit-code
