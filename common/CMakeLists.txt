get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)

include_directories(${CHAINER_COMPILER_ROOT_DIR})
add_library(chainer_compiler_common
  log.cc
  strutil.cc
  )

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(common_test
  strutil_test.cc
  )
target_link_libraries(common_test
  chainer_compiler_common
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  )
add_dependencies(common_test onnx_files)

add_test(
  NAME common_test
  COMMAND common_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
