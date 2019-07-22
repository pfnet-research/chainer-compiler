get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(CHAINER_COMPILER_COMMON_DIR ${CHAINER_COMPILER_ROOT_DIR}/common)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/gsl-lite/include)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_ops.cc
    ${CMAKE_CURRENT_BINARY_DIR}/chxvm.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/chxvm.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_ops.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_chxvm.py
    --input-dir ${CMAKE_CURRENT_SOURCE_DIR}
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
    --protoc-exe ${PROTOC_EXECUTABLE}
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/gen_chxvm.py
  DEPENDS
    ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py
    ${CMAKE_CURRENT_SOURCE_DIR}/chxvm_defs.py
    ${CMAKE_CURRENT_SOURCE_DIR}/chxvm.proto.tmpl
    protoc
    )
add_custom_target(runtime_chxvm_pb_h DEPENDS chxvm.pb.h)

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${OPTIONALLITE_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CHAINER_COMPILER_DLDT_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_NGRAPH_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_TVM_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
add_library(chainer_compiler_runtime
  ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_ops.cc
  ${CMAKE_CURRENT_BINARY_DIR}/chxvm.pb.cc
  backward_context.cc
  chainerx_util.cc
  chrome_tracing.cc
  chxvm.cc
  chxvm_op.cc
  chxvm_state.cc
  chxvm_var.cc
  meminfo.cc
  npy.cc
  ops/activation.cc
  ops/connection.cc
  ops/controlflow.cc
  ops/creation.cc
  ops/cudnn_rnn.cc
  ops/dldt.cc
  ops/generic.cc
  ops/indexing.cc
  ops/logic.cc
  ops/manipulation.cc
  ops/math.cc
  ops/ngraph.cc
  ops/noise.cc
  ops/normalization.cc
  ops/nvrtc.cc
  ops/pooling.cc
  ops/quantize.cc
  ops/resize.cc
  ops/roi.cc
  ops/rnn.cc
  ops/sequence.cc
  ops/something.cc
  ops/sorting.cc
  ops/space_depth.cc
  ops/statistics.cc
  ops/tvm.cc
  )
add_dependencies(
  chainer_compiler_runtime
  runtime_chxvm_pb_h gen_onnx_proto
  )
set_hidden_(chainer_compiler_runtime)

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(chainer_compiler_runtime_test
  npy_test.cc
  chxvm_test.cc
  )
target_link_libraries(chainer_compiler_runtime_test
  chainer_compiler_runtime
  chainer_compiler_compiler
  chainer_compiler_common
  ${CHAINER_COMPILER_CHAINERX_LIBRARIES}
  onnx_proto
  ${PROTOBUF_LIBRARY}
  gtest
  gtest_main
  pthread
  ${CHAINER_COMPILER_NGRAPH_LIBRARIES}
  ${CHAINER_COMPILER_DLDT_LIBRARIES}
  ${CHAINER_COMPILER_TVM_RUNTIME_LIBRARIES}
  ${CHAINER_COMPILER_CUDA_LIBRARIES}
  absl::variant
  absl::optional
  )

add_test(
  NAME chainer_compiler_runtime_test
  COMMAND chainer_compiler_runtime_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
