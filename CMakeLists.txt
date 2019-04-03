cmake_minimum_required(VERSION 3.4)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

execute_process(COMMAND git submodule update --init)

list (INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(third-party)

# Sets default settings from the associated env-var if existed.
# Usage:
#   set_option_ (FOO "Option FOO")
# is equivalent to:
#   if (DEFINED ENV{FOO})
#       option (FOO "Option FOO" $ENV{FOO})
#   else ()
#       option (FOO "Option FOO" OFF)
#   endif ()
function (set_option_ var_ desc_)
    if (DEFINED ENV{${var_}})
        set (def_ $ENV{${var_}})
        message (STATUS "# ${var_} = $ENV{${var_}}")
    else ()
        set (def_ OFF)
    endif ()
    option (${var_} ${desc_} ${def_})
endfunction ()

function (set_hidden_ lib_)
  set_target_properties(
    ${lib_} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(
    ${lib_} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endfunction ()


set_option_ (CHAINER_COMPILER_BUILD_CUDA    "Build CUDA backend (if CUDA is available)")
set_option_ (CHAINER_COMPILER_ENABLE_CUDNN  "Enable CUDNN")
set_option_ (CHAINER_COMPILER_ENABLE_NVTX   "Enable NVTX")
set_option_ (CHAINER_COMPILER_ENABLE_NVRTC  "Enable NVRTC")
set_option_ (CHAINER_COMPILER_ENABLE_OPENCV "Enable OPENCV")
set_option_ (CHAINER_COMPILER_ENABLE_PYTHON "Enable PYTHON")
set_option_ (CHAINER_COMPILER_ENABLE_TVM    "Enable TVM")
set_option_ (CHAINER_COMPILER_USE_SYSTEM_PROTOBUF "Use protobuf installed in the system")

option(CHAINER_COMPILER_NGRAPH_DIR "The path to ngraph_dist" OFF)

option(CHAINER_COMPILER_BUILD_TESTS "Build C++ tests" ON)
option(CHAINER_COMPILER_GENERATE_TESTS "Generate tests for scripts/runtests.py" ON)

if(CHAINER_COMPILER_GENERATE_TESTS)
  set(CHAINER_COMPILER_TEST_ALL ALL)
else()
  set(CHAINER_COMPILER_TEST_ALL "")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-trigraphs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# ChainerX will be linked dynamically. Build it before setting
# visibility flags.
if (${CHAINER_COMPILER_ENABLE_PYTHON})
  set(CHAINERX_BUILD_PYTHON ON)
endif()
add_subdirectory(third_party/chainer/chainerx_cc)

# TODO(hamaji): Get `set_hidden_(libprotobuf)` work. CMP0063 should do
# the trick, but `include` without NO_POLICY_SCOPE in protobuf's
# CMakeLists.txt seems to prevent it from working.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

if (${CHAINER_COMPILER_USE_SYSTEM_PROTOBUF})
  set(PROTOC_EXECUTABLE protoc)
  set(PROTOBUF_LIBRARY protobuf)
else()
  set(protobuf_BUILD_TESTS OFF)
  add_subdirectory(third_party/protobuf/cmake EXCLUDE_FROM_ALL)
  include_directories(third_party/protobuf/src)
  set(PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/third_party/protobuf/cmake/protoc)
  set(PROTOBUF_LIBRARY libprotobuf)
endif()

set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE})

set(ONNX_ML ON)
set(ONNX_NAMESPACE "chainer_compiler_onnx")
add_subdirectory(third_party/onnx)
add_custom_target(onnx_files
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/third_party/onnx/onnx/onnx_chainer_compiler_onnx-ml.pb.h
  )
# It seems `onnx` target compiles onnx*.pb.cc. As this will let other
# files produce warnings, we suppress all warnings from ONNX by -w.
target_compile_definitions(onnx PRIVATE ONNX_API=)
target_compile_options(onnx PRIVATE -w)
add_dependencies(gen_onnx_proto protoc)
add_dependencies(gen_onnx_operators_proto protoc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/chainer/chainerx_cc)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/chainer/chainerx_cc/gsl-lite/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/chainer/chainerx_cc/optional-lite/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/chainer/chainerx_cc/googletest-src/googletest/include)

enable_testing()

add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=${ONNX_NAMESPACE})

# OpenCV
if(${CHAINER_COMPILER_ENABLE_OPENCV})
  find_package(OpenCV REQUIRED)
endif()

if(POLICY ${CMP0077})
  # Make both chainer-compiler and ChainerX consistent.
  set(CHAINERX_BUILD_CUDA ${CHAINER_COMPILER_BUILD_CUDA})
else()
  if(${CHAINER_COMPILER_BUILD_CUDA})
    if(NOT ${CHAINERX_BUILD_CUDA})
      message(FATAL_ERROR "Please set -DCHAINERX_BUILD_CUDA=ON when -DCHAINER_COMPILER_ENABLE_CUDA is ON")
    endif()
  else()
    if(${CHAINERX_BUILD_CUDA})
      message(FATAL_ERROR "Please set -DCHAINERX_BUILD_CUDA=OFF when -DCHAINER_COMPILER_ENABLE_CUDA is OFF")
    endif()
  endif()
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

if(NOT ("${CHAINER_COMPILER_NGRAPH_DIR}" STREQUAL "OFF"))
  add_definitions(-DCHAINER_COMPILER_ENABLE_NGRAPH=1)
  set(CHAINER_COMPILER_NGRAPH_INCLUDE_DIRS ${CHAINER_COMPILER_NGRAPH_DIR}/include)
  set(CHAINER_COMPILER_NGRAPH_LINK_DIRS ${CHAINER_COMPILER_NGRAPH_DIR}/lib)
  set(CHAINER_COMPILER_NGRAPH_LIBRARIES ngraph)
  link_directories(AFTER ${CHAINER_COMPILER_NGRAPH_LINK_DIRS})
endif()

if(${CHAINER_COMPILER_ENABLE_TVM})
  add_definitions(-DCHAINER_COMPILER_ENABLE_TVM=1)
  set(CHAINER_COMPILER_TVM_INCLUDE_DIRS ${CHAINER_COMPILER_TVM_INCLUDE_DIRS} /usr/local/include/HalideIR)
  set(CHAINER_COMPILER_TVM_LIBRARIES tvm)
  set(CHAINER_COMPILER_TVM_RUNTIME_LIBRARIES tvm_runtime)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx ${CMAKE_CURRENT_BINARY_DIR}/third_party/onnx)

add_custom_target(large_tests)

if (${CHAINER_COMPILER_ENABLE_PYTHON})
  include_directories(${CMAKE_BINARY_DIR}/third_party/chainer/chainerx_cc/pybind11/include)
  add_subdirectory(python)
endif()

set(ONNX_CHAINER_DEPS
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/context.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/export.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/export_testcase.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/mapping.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/onnx_helper.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/activation.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/array.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/connection.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/math.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/noise.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/normalization.py
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx-chainer/onnx_chainer/functions/pooling.py
  )

add_subdirectory(compiler)
add_subdirectory(common)
add_subdirectory(feeder)
add_subdirectory(runtime)
add_subdirectory(scripts)
add_subdirectory(tools)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx_chainer_test_stamp
  COMMAND PYTHONPATH=third_party/onnx-chainer pytest third_party/onnx-chainer/tests/functions_tests && touch ${CMAKE_CURRENT_BINARY_DIR}/onnx_chainer_test_stamp
  DEPENDS
  third_party/onnx-chainer/tests/functions_tests/test_activations.py
  third_party/onnx-chainer/tests/functions_tests/test_arrays.py
  third_party/onnx-chainer/tests/functions_tests/test_connections.py
  third_party/onnx-chainer/tests/functions_tests/test_loss.py
  third_party/onnx-chainer/tests/functions_tests/test_maths.py
  third_party/onnx-chainer/tests/functions_tests/test_noises.py
  third_party/onnx-chainer/tests/functions_tests/test_normalizations.py
  third_party/onnx-chainer/tests/functions_tests/test_poolings.py
  ${ONNX_CHAINER_DEPS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_custom_target(
  onnx_chainer_test
  ALL
  DEPENDS onnx_chainer_test_stamp
  )

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
    ResizeImages
    Roi
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
    Unpooling2D
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

set(ELICHIKA_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/chainer2onnx.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/config.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/core.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/functions.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/functions_builtin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/graphs.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/nodes.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/utils.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/values.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/values_builtin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/veval_bin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/veval_unary.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/vevaluator.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/elichika/parser/visualizer.py

  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/testtools/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/testtools/initializer.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/testtools/test_args.py
  ${CMAKE_CURRENT_SOURCE_DIR}/elichika/testtools/testcasegen.py
  )

# Include elichika_tests so the build file will be regenerated when a
# new test is added.
include(scripts/elichika_tests.py)

function(gen_elichika_test dir all)
  set(elichika_tests_py ${CMAKE_CURRENT_SOURCE_DIR}/scripts/elichika_tests.py)
  set(out_stamp ${CMAKE_CURRENT_BINARY_DIR}/stamp_out/elichika_${dir})

  execute_process(
    COMMAND ${elichika_tests_py} --list ${dir}
    OUTPUT_VARIABLE test_files
    )

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stamp_out)

  add_custom_command(
    OUTPUT ${out_stamp}
    COMMAND PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/elichika python3 ${elichika_tests_py} --generate ${dir} && touch ${out_stamp}
    DEPENDS ${elichika_tests_py} ${test_files} ${ELICHIKA_FILES}
    )

  message(${out_stamp})

  add_custom_target(
    gen_elichika_${dir}
    ${all}
    DEPENDS ${out_stamp})

  add_dependencies(large_tests gen_elichika_${dir})

endfunction()

gen_elichika_test(node "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(syntax "${CHAINER_COMPILER_TEST_ALL}")
