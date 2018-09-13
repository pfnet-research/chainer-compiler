#!/usr/bin/python3

import os
import sys

import chainer
import numpy as np
from onnx import onnx_pb
import onnx_chainer


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oniku.scripts import onnx_chainer_util


class AnyModel(chainer.Chain):
    def __init__(self, fn, params):
        super(AnyModel, self).__init__()
        with self.init_scope():
            for name, value in params.items():
                setattr(self, name, chainer.Parameter(value, name=name))
        self.fn = fn

    def __call__(self, *args):
        result = self.fn(self, *args)
        result.name = 'output'
        return result


def create_backprop_test(test_name, fn, **kwargs):
    test_dir = 'out/backprop_test_oc_%s' % test_name

    params = {}
    for name, value in kwargs.items():
        params[name] = np.array(value, np.float32)
    model = AnyModel(fn, params)

    onnx_chainer_util.create_onnx_test('backprop_test_oc_' + test_name,
                                       model,
                                       (),
                                       __builtins__,
                                       test_dir)


class BackpropTest(object):
    def __init__(self, name, fn, **kwargs):
        self.name = name
        self.fn = fn
        self.kwargs = kwargs

    def generate(self):
        create_backprop_test(self.name, self.fn, **self.kwargs)


def get_backprop_tests():
    F = chainer.functions
    tests = []

    def test(name, fn, **kwargs):
        tests.append(BackpropTest(name, fn, **kwargs))

    def aranges(*shape):
        r = 1
        for d in shape:
            r *= d
        return np.arange(r).reshape(shape).astype(np.float32)

    test('add1', lambda m: m.a + m.b, a=[3], b=[7])
    test('mul1', lambda m: m.a * m.b, a=[3], b=[7])
    test('add', lambda m: m.a + m.b, a=[3, 5], b=[7, 2])
    test('sub', lambda m: m.a - m.b, a=[3, 5], b=[7, 2])
    test('mul', lambda m: m.a * m.b, a=[3, 5], b=[7, 2])
    test('div', lambda m: m.a / m.b, a=[3, 5], b=[7, 2])
    test('neg', lambda m: -m.a, a=[3, 5])
    test('exp', lambda m: F.exp(m.a), a=[3, 5])
    test('sigmoid', lambda m: F.sigmoid(m.a), a=[-4, 3, 5])
    test('relu', lambda m: F.relu(m.a), a=[-3, 3, 5])
    test('reduce_sum', lambda m: F.sum(m.a, axis=0), a=[3, 5, 7])

    test('mul_same', lambda m: m.a * m.a, a=[3, 5])

    test('reshape', lambda m: F.reshape(m.a, (1, 2, 1)), a=[3, 5])

    test('sqrt', lambda m: F.sqrt(m.a), a=[3, 5])
    test('tanh', lambda m: F.tanh(m.a), a=[0.3, 0.6])

    # ONNX chainer creates an extra parameter named 'None' for bias of
    # Gemm.
    test('matmul', lambda m: F.matmul(m.a, m.b),
         a=[[3, 5], [7, 4], [2, 6]], b=[[2, 4, 8, 9], [4, 2, 12, 6]])
    test('matmul_ta', lambda m: F.matmul(m.a, m.b, transa=True),
         a=np.transpose([[3, 5], [7, 4], [2, 6]]),
         b=[[2, 4, 8, 9], [4, 2, 12, 6]])
    test('matmul_tb', lambda m: F.matmul(m.a, m.b, transb=True),
         a=[[3, 5], [7, 4], [2, 6]],
         b=np.transpose([[2, 4, 8, 9], [4, 2, 12, 6]]))
    test('matmul_ta_tb', lambda m: F.matmul(m.a, m.b, transa=True, transb=True),
         a=np.transpose([[3, 5], [7, 4], [2, 6]]),
         b=np.transpose([[2, 4, 8, 9], [4, 2, 12, 6]]))
    test('gemm', lambda m: F.linear(m.a, m.b, b=m.c),
         a=[[3, 5], [7, 4], [2, 6]],
         b=np.transpose([[2, 4, 8, 9], [4, 2, 12, 6]]),
         c=[4, 5, 6, 7])

    test('conv', lambda m: F.convolution_2d(m.a, m.b),
         a=aranges(1, 1, 5, 5),
         b=aranges(1, 1, 3, 3))
    # The 4th parameter is calculating gradients of bias more complex.
    test('conv_bias', lambda m: F.convolution_2d(m.a, m.b, b=m.c) * m.d,
         a=aranges(1, 2, 5, 5),
         b=aranges(4, 2, 3, 3),
         c=aranges(4),
         d=aranges(1, 4, 3, 3))
    test('max_pool', lambda m: F.max_pooling_2d(m.a, 3, stride=1,
                                                cover_all=False) * m.b,
         a=aranges(2, 3, 5, 5) % 9,
         b=aranges(2, 3, 3, 3))
    test('average_pool', lambda m: F.average_pooling_2d(m.a, 3,
                                                        stride=1) * m.b,
         a=aranges(2, 3, 5, 5) % 9,
         b=aranges(2, 3, 3, 3))

    test('log_softmax', lambda m: F.log_softmax(m.a),
         a=[[2, 4, 8], [3, 1, 9], [4, 12, 6]])
    # Multiply `b` to avoid nearly zero gradients.
    test('softmax_axis0', lambda m: F.softmax(m.a, axis=0) * m.b,
         a=[[2, 4, 8], [3, 1, 9], [4, 12, 6]],
         b=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    test('softmax_axis1', lambda m: F.softmax(m.a, axis=1) * m.b,
         a=[[2, 4, 8], [3, 1, 9], [4, 12, 6]],
         b=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    test('batch_normalization',
         lambda m: F.batch_normalization(m.x, m.g, m.b) * m.r,
         x=aranges(2, 5, 3, 3),
         g=aranges(5),
         b=aranges(5),
         r=aranges(2, 5, 3, 3) % 7)

    return tests


def main():
    for test in get_backprop_tests():
        test.generate()
    # TODO(hamaji): Stop writing a file to scripts.
    with open('scripts/backprop_test_oc_stamp', 'w'): pass


if __name__ == '__main__':
    main()
