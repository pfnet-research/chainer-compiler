get_filename_component(ONIKU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(ONIKU_COMMON_DIR ${ONIKU_ROOT_DIR}/common)
set(GOOGLETEST_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/gsl-lite/include)
set(OPTIONALLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/optional-lite/include)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_node.py
  MAIN_DEPENDENCY gen_node.py
  DEPENDS ${ONIKU_COMMON_DIR}/codegen_util.py)
add_custom_target(gen_node_base_h DEPENDS gen_node_base.h)

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${OPTIONALLITE_INCLUDE_DIRS})
include_directories(${ONIKU_ROOT_DIR})
include_directories(${CMAKE_BINARY_DIR}) # the root directory instead of `build/compiler`
add_library(oniku_compiler
  dtype.cc
  dtype_inference.cc
  flags.cc
  fusion.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  gradient.cc
  gradient_ops.cc
  graph.cc
  graph_builder.cc
  memory_simulator.cc
  model.cc
  node.cc
  passes.cc
  recompute.cc
  scheduler.cc
  simplifier.cc
  subgraph_canonicalizer.cc
  tensor.cc
  type.cc
  type_inference.cc
  value.cc
  xcvm_emitter.cc
  )
add_dependencies(
  oniku_compiler
  runtime_xcvm_pb_h runtime_xcvm_proto_util_h gen_node_base_h onnx_files
  )

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(compiler_test
  dtype_inference_test.cc
  gradient_test.cc
  model_test.cc
  scheduler_test.cc
  tensor_test.cc
  xcvm_emitter_test.cc
  )
add_dependencies(compiler_test runtime_xcvm_pb_h)
target_link_libraries(compiler_test
  oniku_compiler
  oniku_runtime
  oniku_common
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  )

add_test(
  NAME compiler_test
  COMMAND compiler_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
