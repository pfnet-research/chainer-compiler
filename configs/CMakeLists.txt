get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)

set(jsons
  chxvm.json
  )
set(json_cs)

foreach(json ${jsons})
  set(out ${CMAKE_CURRENT_BINARY_DIR}/${json}.c)
  add_custom_command(
    OUTPUT ${out}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/json_to_c.py ${CMAKE_CURRENT_SOURCE_DIR}/${json} > ${json}.tmp && mv ${json}.tmp ${out}
    MAIN_DEPENDENCY ${json}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/json_to_c.py
    )
  list(APPEND json_cs ${out})
endforeach()

include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CHAINER_COMPILER_ROOT_DIR}/third_party/json/include)

add_library(
  configs
  ${json_cs}
  backend_config.cc
  )
