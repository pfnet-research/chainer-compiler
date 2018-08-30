include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)

include_directories(${CUDA_INCLUDE_DIRS})

add_library(oniku_tools
  util.cc
  )

add_executable(dump_onnx dump_onnx.cc)
target_link_libraries(dump_onnx
  onnx_proto
  oniku_tools
  oniku_compiler
  oniku_runtime
  oniku_common
  xchainer
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
  oniku_common
  xchainer
  onnx_proto
  protobuf
  ${CUDA_CUDART_LIBRARY}
  )
set_target_properties(run_onnx PROPERTIES OUTPUT_NAME "run_onnx")

add_executable(train_imagenet train_imagenet.cc)
target_link_libraries(train_imagenet
  oniku_tools
  oniku_compiler
  oniku_runtime
  oniku_common
  feeder
  xchainer
  onnx_proto
  protobuf
  pthread
  ${CUDA_CUDART_LIBRARY}
  ${OpenCV_LIBS}
  )
set_target_properties(train_imagenet PROPERTIES OUTPUT_NAME "train_imagenet")
