get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
include_directories(${CHAINER_COMPILER_ROOT_DIR})
include_directories(${CHAINER_COMPILER_ROOT_DIR}/third_party/abseil)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)

foreach(f cxx_args apply_cxx_args pybind_args)
  add_custom_command(
    OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/${f}.inc"
    COMMAND
    ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py" --mode "chainer_compiler_core.${f}.inc"
    --output "${CMAKE_CURRENT_BINARY_DIR}/${f}.inc"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/generate_flags_code.py"
    )
  list(APPEND CHAINER_COMPILER_CC_INC "${CMAKE_CURRENT_BINARY_DIR}/${f}.inc")
endforeach()
add_custom_target(
  chainer_compiler_cc_inc
  DEPENDS ${CHAINER_COMPILER_CC_INC}
  )

pybind11_add_module(_chainer_compiler_core.so MODULE chainer_compiler_core.cc)
add_dependencies(_chainer_compiler_core.so chainer_compiler_cc_inc)
target_link_libraries(_chainer_compiler_core.so
  PRIVATE
  chainer_compiler_tools
  chainer_compiler_compiler
  chainer_compiler_runtime
  chainer_compiler_common
  chainer_compiler_configs
  ${CHAINER_COMPILER_DEPENDENCY_LIBRARIES})
set_target_properties(_chainer_compiler_core.so
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}")
set_hidden_(_chainer_compiler_core.so)
