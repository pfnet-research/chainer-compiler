cmake_minimum_required(VERSION 3.4)

execute_process(COMMAND git submodule update --init)

include(cmake/third-party.cmake)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

if(DEFINED ENV{ONIKU_BUILD_CUDA})
    set(DEFAULT_ONIKU_BUILD_CUDA $ENV{ONIKU_BUILD_CUDA})
else()
    set(DEFAULT_ONIKU_BUILD_CUDA ON)
endif()
option(ONIKU_BUILD_CUDA "Build CUDA backend (if CUDA is available)" ${DEFAULT_ONIKU_BUILD_CUDA})

if(DEFINED ENV{ONIKU_ENABLE_CUDNN})
    set(DEFAULT_ONIKU_ENABLE_CUDNN $ENV{ONIKU_ENABLE_CUDNN})
else()
    set(DEFAULT_ONIKU_ENABLE_CUDNN OFF)
endif()
option(ONIKU_ENABLE_CUDNN "Enable CUDNN" ${DEFAULT_ONIKU_ENABLE_CUDNN})

if(DEFINED ENV{ONIKU_ENABLE_NVTX})
    set(DEFAULT_ONIKU_ENABLE_NVTX $ENV{ONIKU_ENABLE_NVTX})
else()
    set(DEFAULT_ONIKU_ENABLE_NVTX OFF)
endif()
option(ONIKU_ENABLE_NVTX "Enable NVTX" ${DEFAULT_ONIKU_ENABLE_NVTX})

if(DEFINED ENV{ONIKU_ENABLE_NVRTC})
    set(DEFAULT_ONIKU_ENABLE_NVRTC $ENV{ONIKU_ENABLE_NVRTC})
else()
    set(DEFAULT_ONIKU_ENABLE_NVRTC OFF)
endif()
option(ONIKU_ENABLE_NVRTC "Enable NVRTC" ${DEFAULT_ONIKU_ENABLE_NVRTC})

if(DEFINED ENV{ONIKU_ENABLE_OPENCV})
    set(DEFAULT_ONIKU_ENABLE_OPENCV $ENV{ONIKU_ENABLE_OPENCV})
else()
    set(DEFAULT_ONIKU_ENABLE_OPENCV OFF)
endif()
option(ONIKU_ENABLE_OPENCV "Enable OPENCV" ${DEFAULT_ONIKU_ENABLE_OPENCV})

if(DEFINED ENV{ONIKU_ENABLE_PYTHON})
    set(DEFAULT_ONIKU_ENABLE_PYTHON $ENV{ONIKU_ENABLE_PYTHON})
else()
    set(DEFAULT_ONIKU_ENABLE_PYTHON OFF)
endif()
option(ONIKU_ENABLE_PYTHON "Enable PYTHON" ${DEFAULT_ONIKU_ENABLE_PYTHON})

option(ONIKU_GENERATE_TESTS "Generate tests for scripts/runtests.py" ON)

if(${ONIKU_GENERATE_TESTS})
  set(ONIKU_TEST_ALL ALL)
else()
  set(ONIKU_TEST_ALL "")
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
if(${ONIKU_ENABLE_OPENCV})
  find_package(OpenCV REQUIRED)
endif()

# CUDA
if(${ONIKU_BUILD_CUDA})
  find_package(CUDA REQUIRED)
  if(${CUDA_FOUND})
    add_definitions(-DONIKU_ENABLE_CUDA=1)
    set(ONIKU_CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})

    if(${ONIKU_ENABLE_CUDNN})
      add_definitions(-DONIKU_ENABLE_CUDNN=1)
      find_package(CuDNN 7 REQUIRED)
      include_directories(${CUDNN_INCLUDE_DIRS})
      link_directories(${CUDNN_LIBRARY_DIRS})
      set(ONIKU_CUDA_LIBRARIES ${ONIKU_CUDA_LIBRARIES} ${CUDNN_LIBRARIES})
    endif()

    if(${ONIKU_ENABLE_NVRTC})
      add_definitions(-DONIKU_ENABLE_NVRTC=1)
      link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      set(ONIKU_CUDA_LIBRARIES ${ONIKU_CUDA_LIBRARIES} cuda nvrtc)
    endif()

    if(${ONIKU_ENABLE_NVTX})
      add_definitions(-DONIKU_ENABLE_NVTX=1)
      link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      set(ONIKU_CUDA_LIBRARIES ${ONIKU_CUDA_LIBRARIES} nvToolsExt)
    endif()

  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnx ${CMAKE_CURRENT_BINARY_DIR}/onnx)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/onnx)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest)

# gsl-lite
# gsl-lite is a header-only library, we do not need to build and run tests
get_third_party(gsl-lite)
include_directories(${CMAKE_BINARY_DIR}/gsl-lite/include)
# ref. https://github.com/martinmoene/gsl-lite#api-macro
# As default, functions (methods) are decorated with __host__ __device__ for the CUDA platform.
# We want to stop it because we get many warnings with nvcc.
add_definitions(-Dgsl_api=)
if(${CUDA_FOUND})
  list(APPEND CUDA_NVCC_FLAGS "-Dgsl_api=")
endif()

# optional-lite
get_third_party(optional-lite)
include_directories(${CMAKE_BINARY_DIR}/optional-lite/include)

add_custom_target(large_tests)

add_subdirectory(compiler)
add_subdirectory(common)
add_subdirectory(feeder)
add_subdirectory(runtime)
add_subdirectory(scripts)
add_subdirectory(tools)

if (${ONIKU_ENABLE_PYTHON})
  get_third_party(pybind11)
  include_directories(${CMAKE_BINARY_DIR}/pybind11/include)
  add_subdirectory(${CMAKE_BINARY_DIR}/pybind11 ${CMAKE_BINARY_DIR}/pybind11-build)
add_subdirectory(python)
endif()

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

  gen_ch2o_test(node ${ch2o_test} "${ONIKU_TEST_ALL}")

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

  gen_ch2o_test(model ${ch2o_test} "${ONIKU_TEST_ALL}")

endforeach()

foreach(
    ch2o_test
    Alex_with_loss
    GoogleNet_with_loss
    )

  gen_ch2o_test(model ${ch2o_test} "")

endforeach()
