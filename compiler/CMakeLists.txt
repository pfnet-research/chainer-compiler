include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)
add_library(oniku_compiler
  code_emitter.cc
  graph.cc
  model.cc
  node.cc
  scheduler.cc
  tensor.cc
  value.cc
  xchainer_emitter.cc
  )

include_directories(../googletest/googletest/include)
link_directories(../googletest/googletest)
add_executable(compiler_test
  code_emitter_test.cc
  model_test.cc
  scheduler_test.cc
  tensor_test.cc
  )
target_link_libraries(compiler_test
  oniku_compiler
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  )

add_test(NAME compiler_test COMMAND compiler_test)

add_executable(compiler main.cc)
target_link_libraries(compiler oniku_compiler)
target_link_libraries(compiler onnx_proto)
target_link_libraries(compiler protobuf)
