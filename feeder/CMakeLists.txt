include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(..)

add_library(feeder
  data_iterator.cc
  imagenet_iterator.cc
  )

include_directories(../googletest/googletest/include)
add_executable(feeder_test
  data_iterator_test.cc
  imagenet_iterator_test.cc
  )
target_link_libraries(feeder_test
  feeder
  xchainer
  gtest
  gtest_main
  pthread
  # TODO(hamaji): Use FIND_PACKAGE.
  opencv_imgcodecs
  opencv_core
  )

add_test(NAME feeder_test COMMAND feeder_test)
