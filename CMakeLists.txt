cmake_minimum_required(VERSION 3.4)

execute_process(COMMAND git submodule update --init)

include(cmake/third-party.cmake)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

if(DEFINED ENV{CHAINER_COMPILER_BUILD_CUDA})
    set(DEFAULT_CHAINER_COMPILER_BUILD_CUDA $ENV{CHAINER_COMPILER_BUILD_CUDA})
else()
    set(DEFAULT_CHAINER_COMPILER_BUILD_CUDA ON)
endif()
option(CHAINER_COMPILER_BUILD_CUDA "Build CUDA backend (if CUDA is available)" ${DEFAULT_CHAINER_COMPILER_BUILD_CUDA})

if(DEFINED ENV{CHAINER_COMPILER_ENABLE_CUDNN})
    set(DEFAULT_CHAINER_COMPILER_ENABLE_CUDNN $ENV{CHAINER_COMPILER_ENABLE_CUDNN})
else()
    set(DEFAULT_CHAINER_COMPILER_ENABLE_CUDNN OFF)
endif()
option(CHAINER_COMPILER_ENABLE_CUDNN "Enable CUDNN" ${DEFAULT_CHAINER_COMPILER_ENABLE_CUDNN})

if(DEFINED ENV{CHAINER_COMPILER_ENABLE_NVTX})
    set(DEFAULT_CHAINER_COMPILER_ENABLE_NVTX $ENV{CHAINER_COMPILER_ENABLE_NVTX})
else()
    set(DEFAULT_CHAINER_COMPILER_ENABLE_NVTX OFF)
endif()
option(CHAINER_COMPILER_ENABLE_NVTX "Enable NVTX" ${DEFAULT_CHAINER_COMPILER_ENABLE_NVTX})

if(DEFINED ENV{CHAINER_COMPILER_ENABLE_NVRTC})
    set(DEFAULT_CHAINER_COMPILER_ENABLE_NVRTC $ENV{CHAINER_COMPILER_ENABLE_NVRTC})
else()
    set(DEFAULT_CHAINER_COMPILER_ENABLE_NVRTC OFF)
endif()
option(CHAINER_COMPILER_ENABLE_NVRTC "Enable NVRTC" ${DEFAULT_CHAINER_COMPILER_ENABLE_NVRTC})

if(DEFINED ENV{CHAINER_COMPILER_ENABLE_OPENCV})
    set(DEFAULT_CHAINER_COMPILER_ENABLE_OPENCV $ENV{CHAINER_COMPILER_ENABLE_OPENCV})
else()
    set(DEFAULT_CHAINER_COMPILER_ENABLE_OPENCV OFF)
endif()
option(CHAINER_COMPILER_ENABLE_OPENCV "Enable OPENCV" ${DEFAULT_CHAINER_COMPILER_ENABLE_OPENCV})

if(DEFINED ENV{CHAINER_COMPILER_ENABLE_PYTHON})
    set(DEFAULT_CHAINER_COMPILER_ENABLE_PYTHON $ENV{CHAINER_COMPILER_ENABLE_PYTHON})
else()
    set(DEFAULT_CHAINER_COMPILER_ENABLE_PYTHON OFF)
endif()
option(CHAINER_COMPILER_ENABLE_PYTHON "Enable PYTHON" ${DEFAULT_CHAINER_COMPILER_ENABLE_PYTHON})

if(DEFINED ENV{CHAINER_COMPILER_ENABLE_TVM})
    set(DEFAULT_CHAINER_COMPILER_ENABLE_TVM $ENV{CHAINER_COMPILER_ENABLE_TVM})
else()
    set(DEFAULT_CHAINER_COMPILER_ENABLE_TVM OFF)
endif()
option(CHAINER_COMPILER_ENABLE_TVM "Enable TVM" ${DEFAULT_CHAINER_COMPILER_ENABLE_TVM})

option(CHAINER_COMPILER_BUILD_TESTS "Build C++ tests" ON)
option(CHAINER_COMPILER_GENERATE_TESTS "Generate tests for scripts/runtests.py" ON)

if(${CHAINER_COMPILER_GENERATE_TESTS})
  set(CHAINER_COMPILER_TEST_ALL ALL)
else()
  set(CHAINER_COMPILER_TEST_ALL "")
endif()

set(ONNX_ML ON)
set(ONNX_NAMESPACE "oniku_onnx")
add_subdirectory(onnx)
add_custom_target(onnx_files
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/onnx/onnx/onnx_oniku_onnx-ml.pb.h
  )
# It seems `onnx` target compiles onnx*.pb.cc. As this will let other
# files produce warnings, we suppress all warnings from ONNX by -w.
target_compile_definitions(onnx PRIVATE ONNX_API=)
target_compile_options(onnx PRIVATE -w)

add_subdirectory(chainer/chainerx_cc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/chainer/chainerx_cc)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/chainer/chainerx_cc/gsl-lite/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/chainer/chainerx_cc/optional-lite/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/chainer/chainerx_cc/googletest-src/googletest/include)

enable_testing()

add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=${ONNX_NAMESPACE})

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
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/builtin_funcs.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/callable.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/chainer2onnx.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/funcs.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/initializer.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/links.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/test_args.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/testcasegen.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/utils.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/ch2o/value.py
  )

# OpenCV
if(${CHAINER_COMPILER_ENABLE_OPENCV})
  find_package(OpenCV REQUIRED)
endif()

# CUDA
if(${CHAINER_COMPILER_BUILD_CUDA})
  find_package(CUDA REQUIRED)
  if(${CUDA_FOUND})
    add_definitions(-DCHAINER_COMPILER_ENABLE_CUDA=1)
    set(CHAINER_COMPILER_CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})

    if(${CHAINER_COMPILER_ENABLE_CUDNN})
      add_definitions(-DCHAINER_COMPILER_ENABLE_CUDNN=1)
      find_package(CuDNN 7 REQUIRED)
      include_directories(${CUDNN_INCLUDE_DIRS})
      link_directories(${CUDNN_LIBRARY_DIRS})
      set(CHAINER_COMPILER_CUDA_LIBRARIES ${CHAINER_COMPILER_CUDA_LIBRARIES} ${CUDNN_LIBRARIES})
    endif()

    if(${CHAINER_COMPILER_ENABLE_NVRTC})
      add_definitions(-DCHAINER_COMPILER_ENABLE_NVRTC=1)
      link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      set(CHAINER_COMPILER_CUDA_LIBRARIES ${CHAINER_COMPILER_CUDA_LIBRARIES} cuda nvrtc)
    endif()

    if(${CHAINER_COMPILER_ENABLE_NVTX})
      add_definitions(-DCHAINER_COMPILER_ENABLE_NVTX=1)
      link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      set(CHAINER_COMPILER_CUDA_LIBRARIES ${CHAINER_COMPILER_CUDA_LIBRARIES} nvToolsExt)
    endif()

  endif()
endif()

if(${CHAINER_COMPILER_ENABLE_TVM})
  add_definitions(-DCHAINER_COMPILER_ENABLE_TVM=1)
  set(CHAINER_COMPILER_TVM_INCLUDE_DIRS ${CHAINER_COMPILER_TVM_INCLUDE_DIRS} /usr/local/include/HalideIR)
  set(CHAINER_COMPILER_TVM_LIBRARIES tvm)
  set(CHAINER_COMPILER_TVM_RUNTIME_LIBRARIES tvm_runtime)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnx ${CMAKE_CURRENT_BINARY_DIR}/onnx)

add_custom_target(large_tests)

if (${CHAINER_COMPILER_ENABLE_PYTHON})
  get_third_party(pybind11)
  include_directories(${CMAKE_BINARY_DIR}/pybind11/include)
  add_subdirectory(${CMAKE_BINARY_DIR}/pybind11 ${CMAKE_BINARY_DIR}/pybind11-build)
  add_subdirectory(python)
endif()

add_subdirectory(compiler)
add_subdirectory(common)
add_subdirectory(feeder)
add_subdirectory(runtime)
add_subdirectory(scripts)
add_subdirectory(tools)

function(gen_ch2o_test dir ch2o_test all)

  set(ch2o_test_py ${CMAKE_CURRENT_SOURCE_DIR}/ch2o/tests/${dir}/${ch2o_test}.py)
  set(out_dir ${CMAKE_CURRENT_SOURCE_DIR}/out/ch2o_${dir}_${ch2o_test}) # TODO: move to build/ directory
  set(out_stamp ${CMAKE_CURRENT_BINARY_DIR}/stamp_out/ch2o_${dir}_${ch2o_test})

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stamp_out)

  add_custom_command(
    OUTPUT ${out_stamp}
    COMMAND PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/ch2o python3 ${ch2o_test_py} ${out_dir} --quiet && touch ${out_stamp}
    MAIN_DEPENDENCY ${ch2o_test_py}
    DEPENDS ${CH2O_FILES}
    )

  add_custom_target(
    ${dir}_${ch2o_test}
    ${all}
    DEPENDS ${out_stamp})

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

  gen_ch2o_test(syntax ${ch2o_test} "${CHAINER_COMPILER_TEST_ALL}")

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
    Len
    Linear
    Matmul
    MaxPool2d
    Mean
    NStepBiLSTM
    NStepLSTM
    NpArray
    NpFull
    NpZeros
    PadSequence
    Relu
    Reshape
    Separate
    Shape
    Sigmoid
    Size
    SplitAxis
    Squeeze
    Softmax
    SoftmaxClossEntropy
    Stack
    Sum
    SwapAxes
    Tanh
    Variable
    Vstack
    )

  gen_ch2o_test(node ${ch2o_test} "${CHAINER_COMPILER_TEST_ALL}")

endforeach()

foreach(
    ch2o_test
    EspNet_AttDot
    EspNet_AttLoc
    EspNet_BLSTM
    EspNet_Decoder
    EspNet_E2E
    EspNet_VGG2L
    MLP_with_loss
    MyLSTM
    StatelessLSTM

    Resnet_with_loss  # Will not be tested by runtests.py, though.
    )

  gen_ch2o_test(model ${ch2o_test} "${CHAINER_COMPILER_TEST_ALL}")

endforeach()

foreach(
    ch2o_test
    Alex_with_loss
    GoogleNet_with_loss
    )

  gen_ch2o_test(model ${ch2o_test} "")

endforeach()
