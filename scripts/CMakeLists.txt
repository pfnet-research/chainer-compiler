add_custom_command(OUTPUT backprop_test_oc_stamp
                   COMMAND python3 scripts/gen_backprop_tests_oc.py
                   MAIN_DEPENDENCY gen_backprop_tests_oc.py
                   DEPENDS onnx_chainer_util.py ../tools/npz_to_onnx.py
                   WORKING_DIRECTORY ..)

add_custom_command(OUTPUT backprop_test_pc_stamp
                   COMMAND python3 scripts/gen_backprop_tests_pc.py > /dev/null
                   MAIN_DEPENDENCY gen_backprop_tests_pc.py
                   DEPENDS onnx_chainer_util.py
                   ../pc/chainer2onnx.py
                   ../pc/test_args.py
                   ../pc/testcasegen.py
                   WORKING_DIRECTORY ..)

add_custom_command(OUTPUT mnist_mlp_stamp
                   COMMAND python3 scripts/gen_mnist_mlp.py
                   MAIN_DEPENDENCY gen_mnist_mlp.py
                   DEPENDS onnx_chainer_util.py ../tools/npz_to_onnx.py
                   WORKING_DIRECTORY ..)

add_custom_target(gen_test_onnx ALL
                  DEPENDS
                  backprop_test_oc_stamp
                  backprop_test_pc_stamp
                  mnist_mlp_stamp)
