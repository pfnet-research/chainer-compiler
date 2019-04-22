get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(CHAINER_COMPILER_COMMON_DIR ${CHAINER_COMPILER_ROOT_DIR}/common)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/gsl-lite/include)
set(OPTIONALLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/optional-lite/include)

if (${CHAINER_COMPILER_ENABLE_PYTHON})
  add_definitions(-DCHAINER_COMPILER_ENABLE_PYTHON=1)
endif()

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_node.py
  MAIN_DEPENDENCY gen_node.py
  DEPENDS ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py)
add_custom_target(gen_node_base_h DEPENDS gen_node_base.h)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_codegen.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_codegen.cc
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/xcvm/gen_xcvm_codegen.py
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  MAIN_DEPENDENCY xcvm/gen_xcvm_codegen.py
  DEPENDS ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py ${CHAINER_COMPILER_ROOT_DIR}/runtime/xcvm_defs.py)
add_custom_target(
  compiler_xcvm_codegen_h
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_codegen.h
  )

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${OPTIONALLITE_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${CHAINER_COMPILER_TVM_INCLUDE_DIRS})

add_library(chainer_compiler_compiler
  code_emitter.cc
  constant_propagation.cc
  config.cc
  computation_order/core.cc
  computation_order/policy_dummy.cc
  computation_order/policy_chen.cc
  custom_onnx_ops.cc
  dtype.cc
  dtype_inference.cc
  evaluator.cc
  flags.cc
  flops.cc
  fusion.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gen_xcvm_codegen.cc
  gradient.cc
  gradient_ops.cc
  gradient_with_order.cc
  graph.cc
  graph_builder.cc
  memory_simulator.cc
  model.cc
  node.cc
  nvrtc_builder.cc
  passes.cc
  scheduler.cc
  shape_evaluator.cc
  simplifier.cc
  subgraph_canonicalizer.cc
  tensor.cc
  topology.cc
  tvm/compiler.cc
  type.cc
  util.cc
  value.cc
  xcvm/config.cc
  xcvm/emitter.cc
  xcvm/xcvm_value.cc
  )
add_dependencies(
  chainer_compiler_compiler
  runtime_xcvm_pb_h compiler_xcvm_codegen_h gen_node_base_h gen_onnx_proto
  )
set_hidden_(chainer_compiler_compiler)

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(compiler_test
  code_emitter_test.cc
  dtype_inference_test.cc
  evaluator_test.cc
  flops_test.cc
  fusion_test.cc
  gradient_test.cc
  model_test.cc
  scheduler_test.cc
  shape_evaluator_test.cc
  tensor_test.cc
  topology_test.cc
  xcvm/emitter_test.cc
  )
add_dependencies(compiler_test runtime_xcvm_pb_h)
target_link_libraries(compiler_test
  chainer_compiler_compiler
  chainer_compiler_runtime
  chainer_compiler_common
  chainerx
  onnx
  onnx_proto
  ${PROTOBUF_LIBRARY}
  gtest
  gtest_main
  pthread
  ${CHAINER_COMPILER_NGRAPH_LIBRARIES}
  ${CHAINER_COMPILER_TVM_LIBRARIES}
  ${CHAINER_COMPILER_CUDA_LIBRARIES}
  )

add_test(
  NAME compiler_test
  COMMAND compiler_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
