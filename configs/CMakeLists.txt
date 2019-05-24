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

add_library(
  configs
  ${json_cs}
  )
