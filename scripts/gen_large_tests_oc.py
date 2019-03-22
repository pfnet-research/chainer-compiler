#!/usr/bin/python3

import chainer
import numpy as np
import onnx_chainer

import large_models


def create_test(test_name, get_fun, dtype):
    np.random.seed(314)
    chainer.config.dtype = dtype
    model, inputs = get_fun(dtype)

    output_grad = 'backprop' in test_name
    test_dir = 'out/%s' % test_name

    chainer.disable_experimental_feature_warning = True
    onnx_chainer.export_testcase(model,
                                 inputs,
                                 test_dir,
                                 output_grad=output_grad,
                                 train=True,
                                 output_names='loss')


def get_large_tests():
    tests = []

    def test(name, get_fun):
        for dtype in (np.float32, np.float64):
            output_grad = dtype == np.float64
            backprop_str = '_backprop' if output_grad else ''
            test_name = 'large_oc%s_%s_%s' % (backprop_str,
                                              name, dtype.__name__)

            tests.append((test_name, get_fun, dtype))

    test('resnet50', large_models.get_resnet50)

    return tests


def main():
    for test in get_large_tests():
        create_test(*test)


if __name__ == '__main__':
    main()
