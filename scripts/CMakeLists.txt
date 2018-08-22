add_custom_command(OUTPUT backprop_test_stamp
                   COMMAND python3 scripts/gen_backprop_tests.py
                   MAIN_DEPENDENCY gen_backprop_tests.py
                   DEPENDS onnx_chainer_util.py ../tools/npz_to_onnx.py
                   WORKING_DIRECTORY ..)

add_custom_command(OUTPUT mnist_mlp_stamp
                   COMMAND python3 scripts/gen_mnist_mlp.py
                   MAIN_DEPENDENCY gen_mnist_mlp.py
                   DEPENDS onnx_chainer_util.py ../tools/npz_to_onnx.py
                   WORKING_DIRECTORY ..)

add_custom_target(gen_test_onnx ALL
                  DEPENDS backprop_test_stamp mnist_mlp_stamp)
