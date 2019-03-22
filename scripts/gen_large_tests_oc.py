#!/usr/bin/python3

import chainer
import numpy as np
import onnx_chainer

import large_models


def create_backprop_test(test_name, get_fun, dtype):
    test_dir = 'out/large_test_oc_%s' % test_name

    np.random.seed(314)
    chainer.config.dtype = dtype

    model, inputs = get_fun(dtype)
    output_grad = dtype == np.float64

    chainer.disable_experimental_feature_warning = True
    onnx_chainer.export_testcase(model,
                                 inputs,
                                 test_dir,
                                 output_grad=output_grad,
                                 train=True,
                                 output_names='loss')


def get_backprop_tests():
    tests = []

    def test(name, get_fun):
        for dtype in (np.float32, np.float64):
            test_name = '%s_%s' % (name, dtype.__name__)
            tests.append((test_name, get_fun, dtype))

    test('resnet50', large_models.get_resnet50)

    return tests


def main():
    for test in get_backprop_tests():
        create_backprop_test(*test)


if __name__ == '__main__':
    main()
