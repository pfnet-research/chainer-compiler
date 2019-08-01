get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(CHAINER_COMPILER_COMMON_DIR ${CHAINER_COMPILER_ROOT_DIR}/common)
set(GOOGLETEST_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/googletest/googletest/include)
set(GSLLITE_INCLUDE_DIRS ${CHAINER_COMPILER_ROOT_DIR}/gsl-lite/include)

if (${CHAINER_COMPILER_ENABLE_PYTHON})
  add_definitions(-DCHAINER_COMPILER_ENABLE_PYTHON=1)
endif()

add_definitions(-DCHAINER_COMPILER_DLDT_DIR=\"${CHAINER_COMPILER_DLDT_DIR}\")

if(MSVC)
    find_package(PythonInterp 3)
endif()

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_node.py
  MAIN_DEPENDENCY gen_node.py
  DEPENDS ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py)
add_custom_target(gen_node_base_h DEPENDS gen_node_base.h)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.h
    ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.cc
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/chxvm/gen_chxvm_codegen.py
    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  MAIN_DEPENDENCY chxvm/gen_chxvm_codegen.py
  DEPENDS
  ${CHAINER_COMPILER_COMMON_DIR}/codegen_util.py
  ${CHAINER_COMPILER_ROOT_DIR}/runtime/chxvm_defs.py
  ${CHAINER_COMPILER_ROOT_DIR}/runtime/chxvm.proto.tmpl
  )
add_custom_target(
  compiler_chxvm_codegen_h
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.h
  )

add_custom_command(
  OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/flags.h"
  COMMAND
  ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode flags.h
  --output "${CMAKE_CURRENT_BINARY_DIR}/flags.h"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
  )
add_custom_command(
  OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/flags.cc"
  COMMAND
  ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode flags.cc
  --output "${CMAKE_CURRENT_BINARY_DIR}/flags.cc"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
  )
add_custom_target(
  compiler_flags_h
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/flags.h" "${CMAKE_CURRENT_BINARY_DIR}/flags.cc"
  )

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CHAINER_COMPILER_TVM_INCLUDE_DIRS})

add_library(chainer_compiler_compiler
  code_emitter.cc
  constant_propagation.cc
  computation_order/core.cc
  computation_order/policy_chen.cc
  computation_order/policy_custom.cc
  computation_order/policy_dummy.cc
  computation_order/policy_gt.cc
  custom_onnx_ops.cc
  dtype.cc
  dtype_inference.cc
  evaluator.cc
  ${CMAKE_CURRENT_BINARY_DIR}/flags.cc
  flops.cc
  fusion.cc
  fusion_dldt.cc
  fusion_elementwise.cc
  fusion_ngraph.cc
  fusion_tvm.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gen_node_base.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gen_chxvm_codegen.cc
  gradient.cc
  gradient_ops.cc
  gradient_with_order.cc
  graph.cc
  graph_builder.cc
  memory_simulator.cc
  merge.cc
  model.cc
  node.cc
  nvrtc_builder.cc
  onnx.cc
  passes.cc
  scheduler.cc
  shape_evaluator.cc
  simplifier.cc
  subgraph_canonicalizer.cc
  tensor.cc
  topology.cc
  tvm/compiler.cc
  type.cc
  util.cc
  value.cc
  chxvm/emitter.cc
  chxvm/chxvm_value.cc
  )
add_dependencies(
  chainer_compiler_compiler
  runtime_chxvm_pb_h compiler_chxvm_codegen_h gen_node_base_h compiler_flags_h
  gen_onnx_proto gen_onnx_operators_proto
  )
set_hidden_(chainer_compiler_compiler)

# Download and extract test data
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../data/mnist/model.onnx"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  COMMAND
  mkdir -p data && cd data &&
  wget --continue https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz &&
  tar -xvzf mnist.tar.gz
  )
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../data/resnet50/model.onnx"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  COMMAND
  mkdir -p data && cd data &&
  wget --continue https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz &&
  tar -xvzf resnet50.tar.gz
  )
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../data/tiny_yolov2/model.onnx"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  COMMAND
  mkdir -p data && cd data &&
  wget --continue https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz &&
  tar -xvzf tiny_yolov2.tar.gz &&
  mv tiny_yolov2/Model.onnx tiny_yolov2/model.onnx
  )

add_custom_target(
  test_models
  DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../data/mnist/model.onnx"
  "${CMAKE_CURRENT_SOURCE_DIR}/../data/resnet50/model.onnx"
  "${CMAKE_CURRENT_SOURCE_DIR}/../data/tiny_yolov2/model.onnx"
  )

include_directories(${GOOGLETEST_INCLUDE_DIRS})
add_executable(chainer_compiler_compiler_test
  code_emitter_test.cc
  custom_onnx_ops_test.cc
  dtype_inference_test.cc
  evaluator_test.cc
  flops_test.cc
  fusion_test.cc
  gradient_test.cc
  merge_test.cc
  model_test.cc
  scheduler_test.cc
  shape_evaluator_test.cc
  simplifier_test.cc
  tensor_test.cc
  topology_test.cc
  chxvm/emitter_test.cc
  )
add_dependencies(
  chainer_compiler_compiler_test
  runtime_chxvm_pb_h compiler_chxvm_codegen_h gen_node_base_h compiler_flags_h
  gen_onnx_proto test_models
  )
target_link_libraries(chainer_compiler_compiler_test
  chainer_compiler_compiler
  chainer_compiler_runtime
  chainer_compiler_common
  chainer_compiler_configs
  ${CHAINER_COMPILER_CHAINERX_LIBRARIES}
  onnx
  onnx_proto
  ${PROTOBUF_LIBRARY}
  gtest
  gtest_main
  pthread
  ${CHAINER_COMPILER_NGRAPH_LIBRARIES}
  ${CHAINER_COMPILER_DLDT_LIBRARIES}
  ${CHAINER_COMPILER_TVM_LIBRARIES}
  ${CHAINER_COMPILER_CUDA_LIBRARIES}
  absl::variant
  absl::optional
  )

add_test(
  NAME chainer_compiler_compiler_test
  COMMAND chainer_compiler_compiler_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
