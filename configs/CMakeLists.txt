get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)

set(jsons
  chxvm
  chxvm_test
  dldt
  ngraph
  nvrtc
  tvm
  )
set(json_cs)

foreach(json ${jsons})
  set(json "${json}.json")
  set(out ${CMAKE_CURRENT_BINARY_DIR}/${json}.cc)
  add_custom_command(
    OUTPUT ${out}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/json_to_cc.py ${CMAKE_CURRENT_SOURCE_DIR}/${json} > ${json}.tmp && mv ${json}.tmp ${out}
    MAIN_DEPENDENCY ${json}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/json_to_cc.py
    )
  list(APPEND json_cs ${out})
endforeach()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/load_json_from_name.cc
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_load_json_from_name.py ${jsons} > ${CMAKE_CURRENT_BINARY_DIR}/load_json_from_name.cc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_load_json_from_name.py ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
  )

include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CHAINER_COMPILER_ROOT_DIR}/third_party/json/include)

add_library(
  chainer_compiler_configs
  ${json_cs}
  backend_config.cc
  json_repository.cc
  ${CMAKE_CURRENT_BINARY_DIR}/load_json_from_name.cc
  )

add_executable(
  chainer_compiler_configs_test
  backend_config_test.cc
  )

target_link_libraries(
  chainer_compiler_configs_test
  chainer_compiler_configs
  chainer_compiler_common
  gtest
  gtest_main
  )

add_test(
  NAME chainer_compiler_configs_test
  COMMAND chainer_compiler_configs_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
