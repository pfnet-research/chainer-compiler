include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)
add_library(oniku_compiler
  graph.cc
  model.cc
  node.cc
  tensor.cc
  value.cc
  )

include_directories(../googletest/googletest/include)

link_directories(../googletest/googletest)
add_executable(compiler_test
  model_test.cc
  )
target_link_libraries(compiler_test
  oniku_common
  oniku_compiler
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  )

add_test(NAME compiler_test COMMAND compiler_test)
