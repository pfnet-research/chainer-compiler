get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CHAINER_COMPILER_ROOT_DIR}/third_party/abseil)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)

add_library(_chainer_compiler_core.so MODULE chainer_compiler_core.cc)
target_link_libraries(_chainer_compiler_core.so
  PRIVATE
  chainer_compiler_tools
  chainer_compiler_compiler
  chainer_compiler_runtime
  chainer_compiler_common
  chainer_compiler_configs
  ${CHAINER_COMPILER_CHAINERX_LIBRARIES}
  onnx
  onnx_proto
  ${PROTOBUF_LIBRARY}
  pthread
  ${CHAINER_COMPILER_NGRAPH_LIBRARIES}
  ${CHAINER_COMPILER_DLDT_LIBRARIES}
  ${CHAINER_COMPILER_TVM_LIBRARIES}
  ${CHAINER_COMPILER_CUDA_LIBRARIES}
  absl::variant
  )
set_target_properties(_chainer_compiler_core.so
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}")
set_hidden_(_chainer_compiler_core.so)
