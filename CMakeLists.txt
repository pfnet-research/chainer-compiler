cmake_minimum_required(VERSION 3.4)

execute_process(COMMAND git submodule update --init)

option(ONIKU_GENERATE_TESTS "Generate tests for scripts/runtests.py" ON)

if(${ONIKU_GENERATE_TESTS})
  set(ONIKU_TEST_ALL ALL)
else()
  set(ONIKU_TEST_ALL "")
endif()

enable_testing()

add_definitions(-DONNX_ML=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-trigraphs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CH2O_FILES
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/__init__.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/builtin_funcs.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/callable.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/chainer2onnx.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/funcs.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/initializer.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/links.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/test_args.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/testcasegen.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/utils.py
  ${CMAKE_SOURCE_DIR}/ch2o/ch2o/value.py
  )

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(onnx)
include_directories(onnx/build)

link_directories(onnx/build)
link_directories(googletest/googletest)

add_custom_target(large_tests)

add_subdirectory(compiler)
add_subdirectory(common)
add_subdirectory(feeder)
add_subdirectory(runtime)
add_subdirectory(scripts)
add_subdirectory(tools)

function(gen_ch2o_test dir ch2o_test all)

  set(out_dir out/ch2o_${dir}_${ch2o_test})

  add_custom_command(
    OUTPUT ${out_dir}/model.onnx
    COMMAND PYTHONPATH=ch2o python3 ch2o/tests/${dir}/${ch2o_test}.py ${out_dir} --quiet
    MAIN_DEPENDENCY ch2o/tests/${dir}/${ch2o_test}.py
    DEPENDS ${CH2O_FILES}
    )

  add_custom_target(
    ${dir}_${ch2o_test}
    ${all}
    DEPENDS ${out_dir}/model.onnx)

  add_dependencies(large_tests ${dir}_${ch2o_test})

endfunction()

foreach(
    ch2o_test
    Cmp
    For
    ForAndIf
    If
    LinkInFor
    ListComp
    MultiClass
    MultiFunction
    Range
    Sequence
    Slice
    UserDefinedFunc
    )

  gen_ch2o_test(syntax ${ch2o_test} "${ONIKU_TEST_ALL}")

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
    Hstack
    Id
    LRN
    Linear
    MaxPool2d
    NpArray
    NStepBiLSTM
    NStepLSTM
    PadSequence
    Relu
    Reshape
    Sigmoid
    SoftmaxClossEntropy
    SplitAxis
    SwapAxes
    Tanh
    Vstack
    )

  gen_ch2o_test(node ${ch2o_test} "${ONIKU_TEST_ALL}")

endforeach()

foreach(
    ch2o_test
    MLP_with_loss
    MyLSTM
    )

  gen_ch2o_test(model ${ch2o_test} "${ONIKU_TEST_ALL}")

endforeach()

foreach(
    ch2o_test
    Alex_with_loss
    GoogleNet_with_loss
    Resnet_with_loss
    )

  gen_ch2o_test(model ${ch2o_test} "")

endforeach()
