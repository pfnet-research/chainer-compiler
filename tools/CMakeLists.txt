get_filename_component(ONIKU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(GSLLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/gsl-lite/include)
set(OPTIONALLITE_INCLUDE_DIRS ${ONIKU_ROOT_DIR}/optional-lite/include)

include_directories(${GSLLITE_INCLUDE_DIRS})
include_directories(${OPTIONALLITE_INCLUDE_DIRS})
include_directories(${ONIKU_ROOT_DIR})
include_directories(${CMAKE_BINARY_DIR}) # the root directory instead of `build/tools`

include_directories(${CUDA_INCLUDE_DIRS})

add_library(oniku_tools
  compiler_flags.cc
  util.cc
  )
add_dependencies(oniku_tools runtime_xcvm_pb_h)

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

add_library(run_onnx_lib
  run_onnx.cc
  )
add_dependencies(run_onnx_lib runtime_xcvm_pb_h onnx_files)

add_executable(run_onnx run_onnx_main.cc)
target_link_libraries(run_onnx
  run_onnx_lib
  oniku_tools
  oniku_compiler
  oniku_runtime
  oniku_common
  chainerx
  onnx
  onnx_proto
  protobuf
  ${ONIKU_TVM_LIBRARIES}
  ${ONIKU_CUDA_LIBRARIES}
  )
set_target_properties(run_onnx PROPERTIES OUTPUT_NAME "run_onnx")

if(${ONIKU_ENABLE_OPENCV})
  add_executable(train_imagenet train_imagenet.cc)
  add_dependencies(train_imagenet runtime_xcvm_pb_h onnx_files)
  target_link_libraries(train_imagenet
    oniku_tools
    oniku_compiler
    oniku_runtime
    oniku_common
    feeder
    chainerx
    onnx
    onnx_proto
    protobuf
    pthread
    ${ONIKU_TVM_LIBRARIES}
    ${ONIKU_CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    )
  set_target_properties(train_imagenet PROPERTIES OUTPUT_NAME "train_imagenet")
endif()

if (${ONIKU_ENABLE_PYTHON})
  include_directories(${PYTHON_INCLUDE_DIRS})

  add_library(run_onnx_core.so MODULE run_onnx_core.cc)

  target_link_libraries(run_onnx_core.so
    PRIVATE
    run_onnx_lib
    oniku_tools
    oniku_compiler
    oniku_runtime
    oniku_common
    chainerx
    onnx
    onnx_proto
    protobuf
    pthread
    ${ONIKU_TVM_LIBRARIES}
    ${ONIKU_CUDA_LIBRARIES}
    )
  set_target_properties(run_onnx_core.so
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}")
endif()
