add_custom_command(
  OUTPUT gen_node_base.h gen_node_base.cc
  COMMAND python3 gen_node.py
  MAIN_DEPENDENCY gen_node.py
  DEPENDS ../common/codegen_util.py)

include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)
add_library(oniku_compiler
  dtype.cc
  dtype_inference.cc
  flags.cc
  gen_node_base.cc
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
  tensor.cc
  type.cc
  type_inference.cc
  value.cc
  xcvm_emitter.cc
  )

include_directories(../googletest/googletest/include)
add_executable(compiler_test
  dtype_inference_test.cc
  gradient_test.cc
  model_test.cc
  scheduler_test.cc
  tensor_test.cc
  xcvm_emitter_test.cc
  )
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

add_test(NAME compiler_test COMMAND compiler_test)
