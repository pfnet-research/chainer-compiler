get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(CHAINER_COMPILER_COMMON_DIR ${CHAINER_COMPILER_ROOT_DIR}/common)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/gsl-lite/include)
set(OPTIONALLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/optional-lite/include)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_ops.cc
    ${CMAKE_CURRENT_BINARY_DIR}/xcvm.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/xcvm.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_ops.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_xcvm.py
    --input-dir ${CMAKE_CURRENT_SOURCE_DIR}
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/gen_xcvm.py
  DEPENDS
    ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py
    ${CMAKE_CURRENT_SOURCE_DIR}/xcvm_defs.py
    ${CMAKE_CURRENT_SOURCE_DIR}/xcvm.proto.tmpl
    )
add_custom_target(runtime_xcvm_pb_h DEPENDS xcvm.pb.h)

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${OPTIONALLITE_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}) # the root build directory instead of `build/runtime`
add_library(chainer_compiler_runtime
  ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_ops.cc
  ${CMAKE_CURRENT_BINARY_DIR}/xcvm.pb.cc
  backward_context.cc
  chainerx_util.cc
  chrome_tracing.cc
  meminfo.cc
  ops/activation.cc
  ops/connection.cc
  ops/controlflow.cc
  ops/creation.cc
  ops/cudnn_rnn.cc
  ops/space_depth.cc
  ops/generic.cc
  ops/indexing.cc
  ops/logic.cc
  ops/manipulation.cc
  ops/math.cc
  ops/noise.cc
  ops/normalization.cc
  ops/nvrtc.cc
  ops/pooling.cc
  ops/rnn.cc
  ops/sequence.cc
  ops/sorting.cc
  ops/statistics.cc
  ops/tvm.cc
  xcvm.cc
  xcvm_op.cc
  xcvm_state.cc
  xcvm_var.cc
  )
add_dependencies(
  chainer_compiler_runtime
  runtime_xcvm_pb_h onnx_files
  )

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(runtime_test
  xcvm_test.cc
  )
target_link_libraries(runtime_test
  chainer_compiler_runtime
  chainer_compiler_compiler
  chainer_compiler_common
  chainerx
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  ${CHAINER_COMPILER_TVM_RUNTIME_LIBRARIES}
  ${CHAINER_COMPILER_CUDA_LIBRARIES}
  )

add_test(
  NAME runtime_test
  COMMAND runtime_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
