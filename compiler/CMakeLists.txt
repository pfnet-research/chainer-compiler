include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)
add_library(oniku_compiler
  code_emitter.cc
  graph.cc
  model.cc
  node.cc
  passes.cc
  scheduler.cc
  tensor.cc
  value.cc
  xchainer_emitter.cc
  xcvm_emitter.cc
  )

include_directories(../googletest/googletest/include)
link_directories(../googletest/googletest)
add_executable(compiler_test
  code_emitter_test.cc
  model_test.cc
  scheduler_test.cc
  tensor_test.cc
  xcvm_emitter_test.cc
  )
target_link_libraries(compiler_test
  oniku_compiler
  oniku_runtime
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  )

add_test(NAME compiler_test COMMAND compiler_test)

add_executable(onnx_to_xchainer_cc onnx_to_xchainer_cc.cc)
target_link_libraries(onnx_to_xchainer_cc oniku_compiler)
target_link_libraries(onnx_to_xchainer_cc onnx_proto)
target_link_libraries(onnx_to_xchainer_cc protobuf)
