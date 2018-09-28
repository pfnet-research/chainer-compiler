add_custom_command(
  OUTPUT backprop_test_oc_stamp
  COMMAND python3 scripts/gen_backprop_tests_oc.py
  MAIN_DEPENDENCY gen_backprop_tests_oc.py
  DEPENDS onnx_chainer_util.py ../tools/npz_to_onnx.py
  WORKING_DIRECTORY ..
  )

add_custom_command(
  OUTPUT backprop_test_pc_stamp
  COMMAND python3 scripts/gen_backprop_tests_pc.py > /dev/null
  MAIN_DEPENDENCY gen_backprop_tests_pc.py
  DEPENDS onnx_chainer_util.py ${CH2O_FILES}
  WORKING_DIRECTORY ..
  )

add_custom_command(
  OUTPUT extra_test_stamp
  COMMAND python3 scripts/gen_extra_test.py > /dev/null
  MAIN_DEPENDENCY gen_extra_test.py
  DEPENDS oniku_script.py sentiment.py
  WORKING_DIRECTORY ..
  )

add_custom_command(
  OUTPUT mnist_mlp_stamp
  COMMAND python3 scripts/gen_mnist_mlp.py
  MAIN_DEPENDENCY gen_mnist_mlp.py
  DEPENDS onnx_chainer_util.py ../tools/npz_to_onnx.py
  WORKING_DIRECTORY ..
  )

add_custom_target(
  gen_test_onnx ${ONIKU_TEST_ALL}
  DEPENDS
  backprop_test_oc_stamp
  backprop_test_pc_stamp
  extra_test_stamp
  mnist_mlp_stamp
  )
