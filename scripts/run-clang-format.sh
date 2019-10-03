#!/usr/bin/env bash

set -eu

cd $(git rev-parse --show-toplevel)

git ls-files -- '**/*.cc' '**/*.h' '**/*.cpp' '**/*.hpp' | xargs -P4 ${CLANG_FORMAT_BIN:-clang-format} -i
git diff --exit-code
