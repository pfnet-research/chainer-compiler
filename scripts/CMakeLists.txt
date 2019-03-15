get_filename_component(CHAINER_COMPILER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(CHAINER_COMPILER_TOOLS_DIR ${CHAINER_COMPILER_ROOT_DIR}/tools)

function(gen_onnx_by_onnx_chainer_ name_)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name_}_stamp
  COMMAND PYTHONPATH=third_party/onnx-chainer python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_${name_}.py && touch ${CMAKE_CURRENT_BINARY_DIR}/${name_}_stamp
  MAIN_DEPENDENCY gen_${name_}.py
  DEPENDS ${ONNX_CHAINER_DEPS}
  WORKING_DIRECTORY ${CHAINER_COMPILER_ROOT_DIR}
  )
endfunction()

gen_onnx_by_onnx_chainer_(backprop_tests_oc)
gen_onnx_by_onnx_chainer_(mnist_mlp)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/backprop_tests_pc_stamp
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_backprop_tests_pc.py && touch ${CMAKE_CURRENT_BINARY_DIR}/backprop_tests_pc_stamp > /dev/null
  MAIN_DEPENDENCY gen_backprop_tests_pc.py
  DEPENDS ${ONNX_CHAINER_DEPS} ${CH2O_FILES}
  WORKING_DIRECTORY ${CHAINER_COMPILER_ROOT_DIR}
  )

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/extra_test_stamp
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_extra_test.py && touch ${CMAKE_CURRENT_BINARY_DIR}/extra_test_stamp > /dev/null
  MAIN_DEPENDENCY gen_extra_test.py
  DEPENDS onnx_script.py sentiment.py gen_chainercv_test.py chainercv_rpn.py
  WORKING_DIRECTORY ${CHAINER_COMPILER_ROOT_DIR}
  )

add_custom_target(
  gen_test_onnx ${CHAINER_COMPILER_TEST_ALL}
  DEPENDS
  backprop_tests_oc_stamp
  backprop_tests_pc_stamp
  extra_test_stamp
  mnist_mlp_stamp
  )
