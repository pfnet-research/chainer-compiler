get_filename_component(ONIKU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(ONIKU_COMMON_DIR ${ONIKU_ROOT_DIR}/common)
set(GOOGLETEST_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/gsl-lite/include)
set(OPTIONALLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/optional-lite/include)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_ops.cc
    ${CMAKE_CURRENT_BINARY_DIR}/xcvm.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/xcvm.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_codegen.cc
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_codegen.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_ops.h
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_xcvm.py
    --input-dir ${CMAKE_CURRENT_SOURCE_DIR}
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/gen_xcvm.py
  DEPENDS
    ${ONIKU_COMMON_DIR}/codegen_util.py
    ${CMAKE_CURRENT_SOURCE_DIR}/xcvm_defs.py
    ${CMAKE_CURRENT_SOURCE_DIR}/xcvm.proto.tmpl
    )
add_custom_target(runtime_xcvm_pb_h DEPENDS xcvm.pb.h)

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${OPTIONALLITE_INCLUDE_DIRS})
include_directories(${ONIKU_ROOT_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}) # the root build directory instead of `build/runtime`
add_library(oniku_runtime
  ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_ops.cc
  ${CMAKE_CURRENT_BINARY_DIR}/xcvm.pb.cc
  backward_context.cc
  chrome_tracing.cc
  meminfo.cc
  ops/activation.cc
  ops/connection.cc
  ops/constant.cc
  ops/cudnn_rnn.cc
  ops/generic.cc
  ops/logic.cc
  ops/manipulation.cc
  ops/math.cc
  ops/normalization.cc
  ops/nvrtc.cc
  ops/pooling.cc
  ops/rnn.cc
  ops/sequence.cc
  ops/tvm.cc
  xchainer.cc
  xcvm.cc
  xcvm_op.cc
  xcvm_ops.cc
  xcvm_state.cc
  xcvm_var.cc
  )
add_dependencies(
  oniku_runtime
  runtime_xcvm_pb_h onnx_files
  )

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(runtime_test
  xcvm_test.cc
  )
target_link_libraries(runtime_test
  oniku_runtime
  oniku_compiler
  oniku_common
  chainerx
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  ${ONIKU_TVM_RUNTIME_LIBRARIES}
  ${ONIKU_CUDA_LIBRARIES}
  )

add_test(
  NAME runtime_test
  COMMAND runtime_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
