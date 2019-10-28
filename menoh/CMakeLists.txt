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
    set(MENOH_EXAMPLE_JSON "${CMAKE_CURRENT_BINARY_DIR}/menoh_example_default_config.json")
    add_custom_command(
      OUTPUT
        "${MENOH_EXAMPLE_JSON}"
      COMMAND
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode menoh_example_default_config.json
        --output "${CMAKE_CURRENT_BINARY_DIR}/menoh_example_default_config.json"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
      )
    add_custom_target(
      menoh_example_default_config_json
      DEPENDS "${MENOH_EXAMPLE_JSON}"
      )
    add_executable(menoh_example menoh_example.cpp)
    add_dependencies(menoh_example menoh_example_default_config_json)
    target_include_directories(menoh_example PRIVATE
      "${CHAINER_COMPILER_ROOT_DIR}/third_party/json/include"
      "${OpenCV_INCLUDE_DIRS}"
      "${CHAINER_COMPILER_ROOT_DIR}"
    )
    target_link_libraries(menoh_example PRIVATE menoh_chainer_compiler "${OpenCV_LIBRARIES}")
    file(
      DOWNLOAD "https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt"
      "${CMAKE_CURRENT_BINARY_DIR}/synset_words.txt"
      EXPECTED_HASH SHA256=acf75ef0abe89694b19056e0796401068b459c457baa30335f240c7692857355
      SHOW_PROGRESS
      )
    file(
      DOWNLOAD "https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg"
      "${CMAKE_CURRENT_BINARY_DIR}/Light_sussex_hen.jpg"
      EXPECTED_HASH SHA256=fb13d3a43ef3299d2198a7de7b16e9ee8f0939cd41c33bcd26995c22699ed273
      SHOW_PROGRESS
      )
    add_test(
      NAME menoh_example_test
      COMMAND
        menoh_example
        --config "${MENOH_EXAMPLE_JSON}"
        --model "${CMAKE_CURRENT_SOURCE_DIR}/../data/shufflenet/model.onnx"
        --synset_words "${CMAKE_CURRENT_BINARY_DIR}/synset_words.txt"
        --input_image "${CMAKE_CURRENT_BINARY_DIR}/Light_sussex_hen.jpg"
        --input_name "gpu_0/data_0" --output_name "gpu_0/softmax_1"
        --preprocess onnx --top_category "n01514859 hen"
      DEPENDS menoh_example menoh_example_default_config
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

add_executable(run_onnx_menoh run_onnx.cpp menoh_chainer_compiler.cpp)
add_dependencies(run_onnx_menoh menoh_chainer_compiler_cc_inc)
target_include_directories(run_onnx_menoh PRIVATE
  "${CHAINER_COMPILER_ROOT_DIR}/third_party/json/include"
  "${CHAINER_COMPILER_ROOT_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/.."
)
target_link_libraries(run_onnx_menoh PRIVATE ${MENOH_DEP_LIBS})
