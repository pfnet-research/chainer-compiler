get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR})

add_library(chainer_compiler_core.so MODULE oniku.cc)
target_link_libraries(chainer_compiler_core.so
  PRIVATE
  chainer_compiler_tools
  chainer_compiler_compiler
  chainer_compiler_runtime
  chainer_compiler_common
  chainerx
  onnx
  onnx_proto
  protobuf
  pthread
  ${CHAINER_COMPILER_TVM_LIBRARIES}
  ${CHAINER_COMPILER_CUDA_LIBRARIES}
  )
set_target_properties(chainer_compiler_core.so
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}")
