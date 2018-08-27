include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)

# TODO(hamaji): Find CUDA instead of using these fixed paths.
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_library(oniku_tools
  util.cc
  )

add_executable(dump_onnx dump_onnx.cc)
target_link_libraries(dump_onnx
  onnx_proto
  oniku_tools
  oniku_compiler
  protobuf
  )
set_target_properties(dump_onnx PROPERTIES OUTPUT_NAME "dump_onnx")

add_executable(dump_tensor dump_tensor.cc)
target_link_libraries(dump_tensor
  oniku_compiler
  onnx_proto
  protobuf
  )
set_target_properties(dump_tensor PROPERTIES OUTPUT_NAME "dump_tensor")

add_executable(run_onnx run_onnx.cc)
target_link_libraries(run_onnx
  oniku_tools
  oniku_compiler
  oniku_runtime
  xchainer
  onnx_proto
  protobuf
  cudart
  )
set_target_properties(run_onnx PROPERTIES OUTPUT_NAME "run_onnx")

add_executable(train_imagenet train_imagenet.cc)
target_link_libraries(train_imagenet
  oniku_tools
  oniku_compiler
  oniku_runtime
  feeder
  xchainer
  onnx_proto
  protobuf
  cudart
  pthread
  ${OpenCV_LIBS}
  )
set_target_properties(train_imagenet PROPERTIES OUTPUT_NAME "train_imagenet")
