#!/usr/bin/env python3

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

    def test(name, get_fun, kwargs=None):
        # kwargs is used for testing
        for dtype in (np.float32, np.float64):
            output_grad = dtype == np.float64
            backprop_str = '_backprop' if output_grad else ''
            test_name = 'large_oc%s_%s_%s' % (backprop_str,
                                              name, dtype.__name__)

            if kwargs is None:
                kwargs = {}
            tests.append((test_name, get_fun, dtype, kwargs))

    test('resnet50', large_models.get_resnet50)
    test('resnet152', large_models.get_resnet152)
    test('vgg16', large_models.get_vgg16, {'rtol': 2e-2, 'atol': 2e-2})
    test('vgg19', large_models.get_vgg19, {'rtol': 2e-2, 'atol': 2e-2})

    return tests


def main():
    for test_name, get_fun, dtype, _ in get_large_tests():
        create_test(test_name, get_fun, dtype)


if __name__ == '__main__':
    main()
