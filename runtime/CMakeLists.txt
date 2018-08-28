add_custom_command(OUTPUT gen_xcvm_ops.cc xcvm.pb.cc xcvm.pb.h gen_xcvm_ops.h xcvm_proto_util.cc xcvm_proto_util.h
                   COMMAND python3 gen_xcvm.py
                   MAIN_DEPENDENCY gen_xcvm.py
                   DEPENDS ../common/codegen_util.py)

include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)
include_directories(${CUDA_INCLUDE_DIRS})
add_library(oniku_runtime
  gen_xcvm_ops.cc
  meminfo.cc
  ops/normalization.cc
  ops/pooling.cc
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
add_executable(runtime_test
  xcvm_test.cc
  )
target_link_libraries(runtime_test
  oniku_runtime
  oniku_compiler
  xchainer
  onnx_proto
  protobuf
  gtest
  gtest_main
  pthread
  ${CUDA_CUDART_LIBRARY}
  )

add_test(NAME runtime_test COMMAND runtime_test)
