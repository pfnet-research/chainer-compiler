#!/usr/bin/env bash

set -eu

for i in $(git ls-files | grep -e '\.cc$\|\.h$'); do
    clang-format -i $i
done
