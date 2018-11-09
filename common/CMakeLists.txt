get_filename_component(ONIKU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(GOOGLETEST_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/googletest/googletest/include)

include_directories(${ONIKU_ROOT_DIR})
add_library(oniku_common
  strutil.cc
  )

include_directories(${GOOGLETEST_INCLUDE_DIRS})
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
add_dependencies(common_test onnx_files)
