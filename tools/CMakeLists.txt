include_directories(../gsl-lite/include)
include_directories(../optional-lite/include)
include_directories(..)

include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(-DONNX_NAMESPACE=onnx)

add_library(oniku_tools
  compiler_flags.cc
  util.cc
  )

add_executable(dump dump.cc)
target_link_libraries(dump
  onnx_proto
  oniku_tools
  oniku_compiler
  oniku_runtime
  oniku_common
  chainerx
  protobuf
  )
set_target_properties(dump PROPERTIES OUTPUT_NAME "dump")

add_executable(run_onnx run_onnx.cc)
target_link_libraries(run_onnx
  oniku_tools
  oniku_compiler
  oniku_runtime
  oniku_common
  chainerx
  onnx_proto
  onnx
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
  chainerx
  onnx_proto
  onnx
  protobuf
  pthread
  ${CUDA_CUDART_LIBRARY}
  ${OpenCV_LIBS}
  )
set_target_properties(train_imagenet PROPERTIES OUTPUT_NAME "train_imagenet")
