#!/usr/bin/env python3

import shutil

import chainer
import numpy as np
import onnx_chainer

import large_models


def create_test(test_name, get_fun, dtype):
    np.random.seed(314)
    chainer.config.dtype = dtype
    model, inputs = get_fun(dtype)

    if chainer.cuda.available:
        model.to_gpu()
        inputs = [chainer.cuda.to_gpu(i) for i in inputs]

    output_grad = 'backprop' in test_name
    test_dir = 'out/%s' % test_name

    chainer.disable_experimental_feature_warning = True
    shutil.rmtree(test_dir, ignore_errors=True)
    onnx_chainer.export_testcase(model,
                                 inputs,
                                 test_dir,
                                 output_grad=output_grad,
                                 train=True,
                                 output_names='loss')


def get_large_tests():
    tests = []

    def test(name, get_fun, dtype, output_grad, **kwargs):
        backprop_str = '_backprop' if output_grad else ''
        test_name = 'large_oc%s_%s_%s' % (backprop_str,
                                          name, dtype.__name__)
        tests.append((test_name, get_fun, dtype, kwargs))

    # ResNet50 fp32 backprop with a fairly large tolerance.
    test('resnet50', large_models.get_resnet50,
         np.float32, True, rtol=0.3, atol=0.3)
    # ResNet50 fp64 backprop with a small tolerance.
    test('resnet50', large_models.get_resnet50, np.float64, True)
    # ResNet152 fp32 forward.
    test('resnet152', large_models.get_resnet152, np.float32, False)
    # VGG19 fp32 backprop with a fairly large tolerance.
    test('vgg16', large_models.get_vgg16,
         np.float32, True, rtol=0.3, atol=0.3)
    # VGG19 fp32 forward.
    test('vgg19', large_models.get_vgg19, np.float32, False)

    return tests


def main():
    for test_name, get_fun, dtype, _ in get_large_tests():
        create_test(test_name, get_fun, dtype)


if __name__ == '__main__':
    main()
