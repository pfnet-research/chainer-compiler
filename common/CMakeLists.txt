get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)

include_directories(${CHAINER_COMPILER_ROOT_DIR})
add_library(chainer_compiler_common
  log.cc
  strutil.cc
  )
set_hidden_(chainer_compiler_common)

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(chainer_compiler_common_test
  iterator_test.cc
  strutil_test.cc
  )
target_link_libraries(chainer_compiler_common_test
  chainer_compiler_common
  onnx_proto
  ${PROTOBUF_LIBRARY}
  gtest
  gtest_main
  )

if (!WIN32)
  target_link_libraries(chainer_compiler_common_test
  pthread
  )
endif()

add_dependencies(chainer_compiler_common_test gen_onnx_proto)

add_test(
  NAME chainer_compiler_common_test
  COMMAND chainer_compiler_common_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
