get_filename_component(ONIKU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(ONIKU_TOOLS_DIR ${ONIKU_ROOT_DIR}/tools)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/backprop_test_oc_stamp
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_backprop_tests_oc.py && touch ${CMAKE_CURRENT_BINARY_DIR}/backprop_test_oc_stamp
  MAIN_DEPENDENCY gen_backprop_tests_oc.py
  DEPENDS onnx_chainer_util.py ${ONIKU_TOOLS_DIR}/npz_to_onnx.py
  WORKING_DIRECTORY ${ONIKU_ROOT_DIR}
  )

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/backprop_test_pc_stamp
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_backprop_tests_pc.py && touch ${CMAKE_CURRENT_BINARY_DIR}/backprop_test_pc_stamp > /dev/null
  MAIN_DEPENDENCY gen_backprop_tests_pc.py
  DEPENDS onnx_chainer_util.py ${CH2O_FILES}
  WORKING_DIRECTORY ${ONIKU_ROOT_DIR}
  )

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/extra_test_stamp
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_extra_test.py && touch ${CMAKE_CURRENT_BINARY_DIR}/extra_test_stamp > /dev/null
  MAIN_DEPENDENCY gen_extra_test.py
  DEPENDS oniku_script.py sentiment.py
  WORKING_DIRECTORY ${ONIKU_ROOT_DIR}
  )

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mnist_mlp_stamp
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_mnist_mlp.py && touch ${CMAKE_CURRENT_BINARY_DIR}/mnist_mlp_stamp
  MAIN_DEPENDENCY gen_mnist_mlp.py
  DEPENDS onnx_chainer_util.py ${ONIKU_TOOLS_DIR}/npz_to_onnx.py
  WORKING_DIRECTORY ${ONIKU_ROOT_DIR}
  )

add_custom_target(
  gen_test_onnx ${ONIKU_TEST_ALL}
  DEPENDS
  backprop_test_oc_stamp
  backprop_test_pc_stamp
  extra_test_stamp
  mnist_mlp_stamp
  )
