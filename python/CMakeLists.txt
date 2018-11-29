include_directories(${PYTHON_INCLUDE_DIRS})

add_library(oniku.so MODULE oniku.cc)
target_link_libraries(oniku.so
  PRIVATE
  oniku_compiler
  oniku_runtime
  oniku_common
  chainerx
  onnx
  onnx_proto
  protobuf
  pthread
  ${ONIKU_CUDA_LIBRARIES}
  )
set_target_properties(oniku.so
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}")
