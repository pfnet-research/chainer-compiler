cmake_minimum_required(VERSION 3.4)

execute_process(COMMAND git submodule update --init)

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-trigraphs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_directories(onnx/.setuptools-cmake-build)
link_directories(googletest/googletest)

add_subdirectory(compiler)
add_subdirectory(common)
add_subdirectory(feeder)
add_subdirectory(runtime)
add_subdirectory(scripts)
add_subdirectory(tools)

function(gen_ch2o_test dir ch2o_test)

  set(out_dir out/ch2o_${dir}/${ch2o_test})

  add_custom_command(
    OUTPUT ${out_dir}/model.onnx
    COMMAND PYTHONPATH=pc python3 pc/casetest/${dir}/${ch2o_test}.py --output ${out_dir}/model.onnx --test_data_dir ${out_dir}/test_data_set_0 --quiet
    MAIN_DEPENDENCY pc/casetest/${dir}/${ch2o_test}.py
    DEPENDS
    pc/chainer2onnx/__init__.py
    pc/chainer2onnx/builtin_funcs.py
    pc/chainer2onnx/chainer2onnx.py
    pc/chainer2onnx/funcs.py
    pc/chainer2onnx/initializer.py
    pc/chainer2onnx/links.py
    pc/chainer2onnx/test_args.py
    pc/chainer2onnx/testcasegen.py
    pc/chainer2onnx/utils.py
    )

  add_custom_target(
    ${dir}_${ch2o_test}
    ALL
    DEPENDS ${out_dir}/model.onnx)

endfunction()

foreach(
    ch2o_test
    For
    LinkInFor
    ListComp
    MultiClass
    MultiFunction
    Range
    Sequence
    Slice
    UserDefinedFunc
    )

  gen_ch2o_test(syntax ${ch2o_test})

endforeach()

foreach(
    ch2o_test
    AddMul
    AveragePool2d
    BatchNorm
    BroadcastTo
    Ceil
    Concat
    Convolution2D
    Cumsum
    Dropout
    EmbedID
    ExpandDims
    Id
    LRN
    Linear
    MaxPool2d
    NstepBiLSTM
    NstepLSTM
    PadSequence
    Relu
    Reshape
    Sigmoid
    SoftmaxClossEntropy
    SplitAxis
    SwapAxes
    Tanh
    #Vstack
    )

  gen_ch2o_test(node ${ch2o_test})

endforeach()
