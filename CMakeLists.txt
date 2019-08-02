 cmake_minimum_required(VERSION 3.4)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

execute_process(COMMAND git submodule update --init)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(third-party)

# Sets default settings from the associated env-var if existed.
# Usage:
#   set_option_(FOO "Option FOO" OFF)
# is equivalent to:
#   if(DEFINED ENV{FOO})
#       option(FOO "Option FOO" $ENV{FOO})
#   else()
#       option(FOO "Option FOO" OFF)
#   endif()
function(set_option_ var_ desc_ def_)
  if(DEFINED ENV{${var_}})
    set(def_ $ENV{${var_}})
    message(STATUS "# ${var_} = $ENV{${var_}}")
  endif()
  option(${var_} ${desc_} ${def_})
endfunction()

# Ref: https://github.com/edsiper/cmake-options
macro(set_subdirectory_option_ option value)
  set(${option} ${value} CACHE "" INTERNAL FORCE)
endmacro()

function(set_hidden_ lib_)
  set_target_properties(
    ${lib_} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(
    ${lib_} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endfunction()

set_option_(CHAINER_COMPILER_ENABLE_CUDA   "Enable CUDA" OFF)
set_option_(CHAINER_COMPILER_ENABLE_CUDNN  "Enable CUDNN" OFF)
set_option_(CHAINER_COMPILER_ENABLE_NVTX   "Enable NVTX" ON)
set_option_(CHAINER_COMPILER_ENABLE_NVRTC  "Enable NVRTC" ON)
set_option_(CHAINER_COMPILER_ENABLE_OPENCV "Enable OPENCV" OFF)
set_option_(CHAINER_COMPILER_ENABLE_OPENMP "Enable OPENMP" OFF)
set_option_(CHAINER_COMPILER_ENABLE_PYTHON "Enable PYTHON" OFF)
set_option_(CHAINER_COMPILER_ENABLE_TVM    "Enable TVM" OFF)
set_option_(CHAINER_COMPILER_USE_SYSTEM_PROTOBUF "Use protobuf installed in the system" OFF)

option(CHAINER_COMPILER_PREBUILT_CHAINERX_DIR "The path to prebuilt ChainerX" OFF)
option(CHAINER_COMPILER_NGRAPH_DIR "The path to ngraph_dist" OFF)
option(CHAINER_COMPILER_DLDT_DIR "The path to dldt" OFF)

option(CHAINER_COMPILER_BUILD_TESTS "Build C++ tests" ON)
option(CHAINER_COMPILER_GENERATE_TESTS "Generate tests for scripts/runtests.py" ON)

if(CHAINER_COMPILER_GENERATE_TESTS)
  set(CHAINER_COMPILER_TEST_ALL ALL)
else()
  set(CHAINER_COMPILER_TEST_ALL "")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-trigraphs")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
endif()

set(CMAKE_CXX_STANDARD 14)

find_package(PythonInterp 3)

# Make both chainer-compiler and ChainerX consistent.
set_subdirectory_option_(CHAINERX_BUILD_CUDA ${CHAINER_COMPILER_ENABLE_CUDA})

# ChainerX will be linked dynamically. Build it before setting
# visibility flags.
if(${CHAINER_COMPILER_ENABLE_PYTHON})
  set_subdirectory_option_(CHAINERX_BUILD_PYTHON ON)
endif()
add_subdirectory(third_party/chainer/chainerx_cc EXCLUDE_FROM_ALL)

if(NOT ("${CHAINER_COMPILER_PREBUILT_CHAINERX_DIR}" STREQUAL "OFF"))
  set(CHAINER_COMPILER_CHAINERX_LIBRARIES "${CHAINER_COMPILER_PREBUILT_CHAINERX_DIR}/libchainerx.so")
else()
  set(CHAINER_COMPILER_CHAINERX_LIBRARIES "chainerx")
endif()

# TODO(hamaji): Get `set_hidden_(libprotobuf)` work. CMP0063 should do
# the trick, but `include` without NO_POLICY_SCOPE in protobuf's
# CMakeLists.txt seems to prevent it from working.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

if(${CHAINER_COMPILER_USE_SYSTEM_PROTOBUF})
  set(PROTOC_EXECUTABLE protoc)
  set(PROTOBUF_LIBRARY protobuf)
else()
  set_subdirectory_option_(protobuf_BUILD_TESTS OFF)
  add_subdirectory(third_party/protobuf/cmake EXCLUDE_FROM_ALL)
  include_directories(third_party/protobuf/src)
  set(PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/third_party/protobuf/cmake/protoc)
  set(PROTOBUF_LIBRARY libprotobuf)
endif()

set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE})

set(ONNX_ML ON)
set(ONNX_NAMESPACE "chainer_compiler_onnx")
add_subdirectory(third_party/onnx)
# It seems `onnx` target compiles onnx*.pb.cc. As this will let other
# files produce warnings, we suppress all warnings from ONNX by -w.
target_compile_definitions(onnx PRIVATE ONNX_API=)
target_compile_options(onnx PRIVATE -w)
add_dependencies(gen_onnx_proto protoc)
add_dependencies(gen_onnx_operators_proto protoc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/chainer/chainerx_cc)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/chainer/chainerx_cc/gsl-lite/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/chainer/chainerx_cc/googletest-src/googletest/include)

enable_testing()

add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=${ONNX_NAMESPACE})

# OpenCV
if(${CHAINER_COMPILER_ENABLE_OPENCV})
  find_package(OpenCV REQUIRED)
endif()

if(${CHAINER_COMPILER_ENABLE_OPENMP})
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DCHAINER_COMPILER_ENABLE_OPENMP=1)
  else()
    error("OpenMP not found")
  endif()
endif()

# CUDA
if(${CHAINER_COMPILER_ENABLE_CUDA})
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

    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

    if(${CHAINER_COMPILER_ENABLE_NVRTC})
      add_definitions(-DCHAINER_COMPILER_ENABLE_NVRTC=1)
      set(CHAINER_COMPILER_CUDA_LIBRARIES ${CHAINER_COMPILER_CUDA_LIBRARIES} cuda nvrtc)
    endif()
    if(${CHAINER_COMPILER_ENABLE_NVTX})
      add_definitions(-DCHAINER_COMPILER_ENABLE_NVTX=1)
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

if(NOT ("${CHAINER_COMPILER_DLDT_DIR}" STREQUAL "OFF"))
  add_definitions(-DCHAINER_COMPILER_ENABLE_DLDT=1)
  set(CHAINER_COMPILER_DLDT_INCLUDE_DIRS ${CHAINER_COMPILER_DLDT_DIR}/inference-engine/include)
  set(CHAINER_COMPILER_DLDT_LINK_DIRS ${CHAINER_COMPILER_DLDT_DIR}/inference-engine/bin/intel64/Release/lib)
  set(CHAINER_COMPILER_DLDT_LIBRARIES inference_engine ${CMAKE_DL_LIBS})
  link_directories(AFTER ${CHAINER_COMPILER_DLDT_LINK_DIRS})
endif()

if(${CHAINER_COMPILER_ENABLE_TVM})
  add_definitions(-DCHAINER_COMPILER_ENABLE_TVM=1)
  set(CHAINER_COMPILER_TVM_INCLUDE_DIRS ${CHAINER_COMPILER_TVM_INCLUDE_DIRS} /usr/local/include/HalideIR)
  set(CHAINER_COMPILER_TVM_LINK_DIRS ${CHAINER_COMPILER_TVM_DIR}/lib)
  set(CHAINER_COMPILER_TVM_LIBRARIES tvm)
  set(CHAINER_COMPILER_TVM_RUNTIME_LIBRARIES tvm_runtime)
  link_directories(AFTER ${CHAINER_COMPILER_TVM_LINK_DIRS})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx ${CMAKE_CURRENT_BINARY_DIR}/third_party/onnx)

add_custom_target(large_tests)

if(${CHAINER_COMPILER_ENABLE_PYTHON})
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/chainer/chainerx_cc/pybind11/include)
  add_subdirectory(chainer_compiler_cc)
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

set(ABSEIL_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/third_party/chainer/chainerx_cc/abseil)
include_directories(${ABSEIL_INCLUDE_DIRS})

add_subdirectory(compiler)
add_subdirectory(common)
add_subdirectory(configs)
add_subdirectory(feeder)
add_subdirectory(runtime)
add_subdirectory(scripts)
add_subdirectory(tools)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx_chainer_test_stamp
  COMMAND rm -rf "${CMAKE_CURRENT_SOURCE_DIR}/out/opset*" && PYTHONPATH=third_party/onnx-chainer pytest --value-check-runtime skip third_party/onnx-chainer/tests/functions_tests && touch ${CMAKE_CURRENT_BINARY_DIR}/onnx_chainer_test_stamp
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
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/builtin_funcs.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/callable.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/chainer2onnx.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/funcs.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/initializer.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/links.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/test_args.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/testcasegen.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/utils.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/ch2o/value.py
  )

function(gen_ch2o_test dir ch2o_test all)

  set(ch2o_test_py ${CMAKE_CURRENT_SOURCE_DIR}/testcases/ch2o_tests/${dir}/${ch2o_test}.py)
  set(out_dir ${CMAKE_CURRENT_SOURCE_DIR}/out/ch2o_${dir}_${ch2o_test}) # TODO: move to build/ directory
  set(out_stamp ${CMAKE_CURRENT_BINARY_DIR}/stamp_out/ch2o_${dir}_${ch2o_test})

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stamp_out)

  add_custom_command(
    OUTPUT ${out_stamp}
    COMMAND PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${ch2o_test_py} ${out_dir} --quiet && touch ${out_stamp}
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
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/chainer2onnx.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/functions_builtin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/links_builtin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/onnx_converters.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/canonicalizer.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/config.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/core.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/functions.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/functions_builtin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/functions_ndarray.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/graphs.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/links_builtin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/nodes.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/utils.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/values.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/veval_bin.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/veval_unary.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/vevaluator.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/parser/visualizer.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/testtools/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/testtools/initializer.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/testtools/test_args.py
  ${CMAKE_CURRENT_SOURCE_DIR}/chainer_compiler/elichika/testtools/testcasegen.py
  )

# Include elichika_tests so the build file will be regenerated when a
# new test is added.
include(scripts/elichika_tests.py)

function(gen_elichika_test dir all)
  string(REPLACE "/" "_" name ${dir})
  set(elichika_tests_py ${CMAKE_CURRENT_SOURCE_DIR}/scripts/elichika_tests.py)
  set(out_stamp ${CMAKE_CURRENT_BINARY_DIR}/stamp_out/elichika_${name})

  execute_process(
    COMMAND ${elichika_tests_py} --list ${dir}
    OUTPUT_VARIABLE test_files
    )

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/stamp_out)

  add_custom_command(
    OUTPUT ${out_stamp}
    COMMAND PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${elichika_tests_py} --generate ${dir} && touch ${out_stamp}
    DEPENDS ${elichika_tests_py} ${ELICHIKA_FILES} ${test_files}
    )

  message(${out_stamp})

  add_custom_target(
    gen_elichika_${name}
    ${all}
    DEPENDS ${out_stamp})

  add_dependencies(large_tests gen_elichika_${name})

endfunction()

gen_elichika_test(model "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(node "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(node/ndarray "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(node/Functions "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(node/Functions/Activation "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(node/Links "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(syntax "${CHAINER_COMPILER_TEST_ALL}")
gen_elichika_test(chainercv_model/resnet "${CHAINER_COMPILER_TEST_ALL}")

# Menoh
option(CHAINER_COMPILER_ENABLE_MENOH "Build Menoh API" ON)
if(${CHAINER_COMPILER_ENABLE_MENOH})
    add_subdirectory(menoh)
endif()
