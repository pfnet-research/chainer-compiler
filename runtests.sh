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

onnx_tests=()

if [ $# = "0" ]; then
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_relu )

    onnx_tests+=( onnx/onnx/backend/test/data/node/test_add )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_add_bcast )

    onnx_tests+=( onnx/onnx/backend/test/data/node/test_matmul_2d )
    # terminate called after throwing an instance of 'xchainer::NotImplementedError'
    #   what():  dot does not support rhs operand with ndim > 2
    # onnx_tests+=( onnx/onnx/backend/test/data/node/test_matmul_3d )
    # onnx_tests+=( onnx/onnx/backend/test/data/node/test_matmul_4d )

    onnx_tests+=( onnx/onnx/backend/test/data/node/test_basic_conv_with_padding )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_basic_conv_without_padding )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_conv_with_strides_no_padding )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_conv_with_strides_padding )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_conv_with_strides_and_asymmetric_padding )

    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_1d_default )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_default )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_pads )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_precomputed_pads )
    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_precomputed_same_upper )
    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_precomputed_strides )
    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_same_lower )
    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_same_upper )
    onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_2d_strides )
    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_3d_default )
    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_with_argmax_2d_precomputed_pads )
    #onnx_tests+=( onnx/onnx/backend/test/data/node/test_maxpool_with_argmax_2d_precomputed_strides )
else
    onnx_tests+=( $@ )
fi


mkdir -p out
for onnx in "${onnx_tests[@]}"; do
    onnx_model="${onnx}/model.onnx"
    name=$(echo "${onnx_model}" | sed 's/\//_/g')
    cc="out/${name}.cc"
    exe="out/${name}.exe"

    echo "${onnx}..."
    ./compiler/compiler "${onnx_model}" > "${cc}"
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
        args=""
        for a in $(echo "${test_data_set}/input_*.pb"); do
            args="${args} -in ${a}"
        done
        for a in $(echo "${test_data_set}/output_*.pb"); do
            args="${args} -out ${a}"
        done
        "./${exe}" ${args}
    done
done
