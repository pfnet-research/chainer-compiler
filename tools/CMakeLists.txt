include_directories(../onnx/.setuptools-cmake-build)
include_directories(..)

add_executable(dump_onnx dump_onnx.cc)
target_link_libraries(dump_onnx onnx_proto)
target_link_libraries(dump_onnx protobuf)
set_target_properties(dump_onnx PROPERTIES OUTPUT_NAME "dump_onnx")

add_executable(dump_tensor dump_tensor.cc)
target_link_libraries(dump_tensor onnx_proto)
target_link_libraries(dump_tensor protobuf)
set_target_properties(dump_tensor PROPERTIES OUTPUT_NAME "dump_tensor")
