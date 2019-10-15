get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/json_args.inc"
  COMMAND
    ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode menoh_chainer_compiler.json_args.inc
    --output "${CMAKE_CURRENT_BINARY_DIR}/json_args.inc"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
  )
add_custom_command(
  OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/args_json.inc"
  COMMAND
    ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode menoh_chainer_compiler.args_json.inc
    --output "${CMAKE_CURRENT_BINARY_DIR}/args_json.inc"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
  )
add_custom_target(
  menoh_chainer_compiler_cc_inc
  DEPENDS
  "${CMAKE_CURRENT_BINARY_DIR}/args_json.inc"
  "${CMAKE_CURRENT_BINARY_DIR}/json_args.inc"
  )
add_library(menoh_chainer_compiler SHARED menoh_chainer_compiler.cpp menoh.h)
add_dependencies(menoh_chainer_compiler menoh_chainer_compiler_cc_inc)
target_include_directories(menoh_chainer_compiler PRIVATE
  "${CHAINER_COMPILER_ROOT_DIR}/third_party/json/include"
  "${CHAINER_COMPILER_ROOT_DIR}"
  "${PYTHON_INCLUDE_DIRS}"
  "${CMAKE_CURRENT_BINARY_DIR}/.."
)
set(MENOH_DEP_LIBS
  chainer_compiler_tools
  chainer_compiler_compiler
  chainer_compiler_configs
  chainer_compiler_runtime
  chainer_compiler_common
  ${CHAINER_COMPILER_DEPENDENCY_LIBRARIES})
target_link_libraries(menoh_chainer_compiler PRIVATE ${MENOH_DEP_LIBS})

if(OpenCV_FOUND)
    add_custom_command(
      OUTPUT
        "${CMAKE_CURRENT_BINARY_DIR}/menoh_example_default_config.json"
      COMMAND
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode menoh_example_default_config.json
        --output "${CMAKE_CURRENT_BINARY_DIR}/menoh_example_default_config.json"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
      )
    add_custom_target(
      menoh_example_default_config_json
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/menoh_example_default_config.json"
      )
    add_executable(menoh_example menoh_example.cpp)
    add_dependencies(menoh_example menoh_example_default_config_json)
    target_include_directories(menoh_example PRIVATE
      "${CHAINER_COMPILER_ROOT_DIR}/third_party/json/include"
      "${OpenCV_INCLUDE_DIRS}"
      "${CHAINER_COMPILER_ROOT_DIR}"
    )
    target_link_libraries(menoh_example PRIVATE menoh_chainer_compiler "${OpenCV_LIBRARIES}")
endif()

add_executable(run_onnx_menoh run_onnx.cpp menoh_chainer_compiler.cpp)
add_dependencies(run_onnx_menoh menoh_chainer_compiler_cc_inc)
target_include_directories(run_onnx_menoh PRIVATE
  "${CHAINER_COMPILER_ROOT_DIR}/third_party/json/include"
  "${CHAINER_COMPILER_ROOT_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/.."
)
target_link_libraries(run_onnx_menoh PRIVATE ${MENOH_DEP_LIBS})
