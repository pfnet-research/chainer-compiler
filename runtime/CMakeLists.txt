add_custom_command(OUTPUT init_xcvm_ops.cc xcvm.pb.cc xcvm.pb.h xcvm_ops.h xcvm_proto_util.cc xcvm_proto_util.h

                   COMMAND python3 gen_xcvm.py
                   MAIN_DEPENDENCY gen_xcvm.py)

include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)
add_library(oniku_xchainer_runtime
  init_xcvm_ops.cc
  xchainer.cc
  xcvm.cc
  xcvm.pb.cc
  xcvm_ops.cc
  xcvm_state.cc
  xcvm_proto_util.cc
  )

add_library(oniku_xchainer_runtime_main
  xchainer_main.cc
  )

include_directories(../googletest/googletest/include)
link_directories(../googletest/googletest)
add_executable(runtime_test
  xcvm_test.cc
  )
target_link_libraries(runtime_test
  oniku_xchainer_runtime
  xchainer
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  )

add_test(NAME runtime_test COMMAND runtime_test)
