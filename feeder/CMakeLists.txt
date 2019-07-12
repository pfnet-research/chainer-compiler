get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/gsl-lite/include)

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

set(FEEDER_SRCS data_iterator.cc)
set(FEEDER_TEST_SRCS data_iterator_test.cc)
if(${CHAINER_COMPILER_ENABLE_OPENCV})
  set(FEEDER_SRCS ${FEEDER_SRCS} imagenet_iterator.cc)
  set(FEEDER_TEST_SRCS ${FEEDER_TEST_SRCS} imagenet_iterator_test.cc)
endif()

add_library(feeder ${FEEDER_SRCS})
set_hidden_(feeder)

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(chainer_compiler_feeder_test ${FEEDER_TEST_SRCS})
target_link_libraries(chainer_compiler_feeder_test
  feeder
  chainer_compiler_common
  ${CHAINER_COMPILER_CHAINERX_LIBRARIES}
  gtest
  gtest_main
  pthread
  ${OpenCV_LIBS}
  ${CHAINER_COMPILER_CUDA_LIBRARIES}
  )

add_test(
  NAME chainer_compiler_feeder_test
  COMMAND chainer_compiler_feeder_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
