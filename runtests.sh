#!/bin/bash
#
# TODO(hamaji): Probably better to rewrite the test harness by cmake.
#

set -eu

if [ -e Makefile ]; then
    make -j4
elif [ -e build.ninja ]; then
    ninja
fi

CXX=c++

onnx_tests=onnx/onnx/backend/test/data/node/test_relu

mkdir -p out
for onnx in "${onnx_tests}"; do
    onnx_model="${onnx}/model.onnx"
    name=$(echo "${onnx_model}" | sed 's/\//_/g')
    cc="out/${name}.cc"
    exe="out/${name}.exe"

    echo "${onnx}..."
    ./compiler/compiler "${onnx}" > "${cc}"
    "${CXX}" \
        -g -I. \
        -Igsl-lite/include \
        -Ioptional-lite/include \
        -Ionnx/.setuptools-cmake-build \
        runtime/CMakeFiles/oniku_xchainer_runtime_main.dir/xchainer_main.cc.o \
        "${cc}" \
        runtime/liboniku_xchainer_runtime.a \
        compiler/liboniku_compiler.a \
        -lonnx_proto -lxchainer -lprotobuf\
        -o "${exe}"

    for test_data_set in $(echo "${onnx}/test_data_set_*"); do
        inputs=$(echo "${test_data_set}/input_*.pb")
        outputs=$(echo "${test_data_set}/output_*.pb")

        args=""
        for a in "${inputs}"; do
            args="${args} -in ${a}"
        done
        for a in "${outputs}"; do
            args="${args} -out ${a}"
        done
        "./${exe}" ${args}
    done
done
