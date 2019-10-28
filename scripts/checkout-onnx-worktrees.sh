#!/bin/bash

set -eux

cd "$(dirname $0)/../third_party/onnx"

OPSET_RELEASE_TABLE=("11:v1.6.0" "10:v1.5.0" "9:v1.4.1" "8:v1.3.0")

for opset in "${OPSET_RELEASE_TABLE[@]}" ; do
    opset_version="${opset%%:*}"
    onnx_release="${opset##*:}"

    echo "cloning $opset"

    worktree_dir=../onnx-$opset_version
    if [ -d "$worktree_dir" ] ; then
        rm -rf $worktree_dir
    fi

    git clone -b $onnx_release . $worktree_dir
done
