include_directories(..)
add_library(oniku_common
  strutil.cc
  )

include_directories(../googletest/googletest/include)
add_executable(common_test
  strutil_test.cc
  )
target_link_libraries(common_test
  oniku_common
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  )
