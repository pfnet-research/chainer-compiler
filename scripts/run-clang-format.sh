#!/usr/bin/env bash

set -eu

git ls-files | grep -e '\.cc$\|\.h$' | xargs -P4 clang-format -i
git diff --exit-code
