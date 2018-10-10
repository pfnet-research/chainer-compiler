get_filename_component(ONIKU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(GOOGLETEST_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/gsl-lite/include)
set(OPTIONALLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/optional-lite/include)

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${OPTIONALLITE_INCLUDE_DIRS})
include_directories(${ONIKU_ROOT_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(feeder
  data_iterator.cc
  imagenet_iterator.cc
  )

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(feeder_test
  data_iterator_test.cc
  imagenet_iterator_test.cc
  )
target_link_libraries(feeder_test
  feeder
  chainerx
  gtest
  gtest_main
  pthread
  ${OpenCV_LIBS}
  )

add_test(NAME feeder_test COMMAND feeder_test)
