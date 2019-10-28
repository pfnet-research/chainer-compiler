#!/usr/bin/env python3

import shutil

import chainer
import numpy as np
import onnx_chainer

from test_case import TestCase


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


def create_backprop_test(test_name, fn, args, dtype=np.float32, **kwargs):
    test_dir = 'out/%s' % test_name

    params = {}
    for name, value in kwargs.items():
        params[name] = np.array(value, dtype)
    model = AnyModel(fn, params)

    chainer.disable_experimental_feature_warning = True
    shutil.rmtree(test_dir, ignore_errors=True)
    onnx_chainer.export_testcase(model,
                                 args,
                                 test_dir,
                                 output_grad=True,
                                 output_names='loss')


class BackpropTest(TestCase):
    def __init__(self, name, fn, args=(),
                 rtol=None, test_params=None, **kwargs):
        name = 'backprop_test_oc_%s' % name
        test_params = {} if test_params is None else test_params
        super().__init__(basedir='out', name=name, rtol=rtol, **test_params)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def generate(self):
        create_backprop_test(self.name, self.fn, self.args, **self.kwargs)


def get_backprop_tests():
    F = chainer.functions
    tests = []

    def test(name, fn, **kwargs):
        for dtype in (np.float16, np.float32, np.float64):
            test_name = '%s_%s' % (name, dtype.__name__)
            rtol = None if dtype != np.float16 else 0.02
            tests.append(BackpropTest(test_name, fn, dtype=dtype, rtol=rtol,
                                      **kwargs))

    def aranges(*shape):
        r = np.prod(shape)
        return np.arange(r).reshape(shape).astype(np.float32)

    test('add1', lambda m: m.a + m.b, a=[3], b=[7])
    test('mul1', lambda m: m.a * m.b, a=[3], b=[7])
    test('add', lambda m: m.a + m.b, a=[3, 5], b=[7, 2])
    test('sub', lambda m: m.a - m.b, a=[3, 5], b=[7, 2])
    test('mul', lambda m: m.a * m.b, a=[3, 5], b=[7, 2])
    test('div', lambda m: m.a / m.b, a=[3, 5], b=[7, 2])
    test('neg', lambda m: -m.a, a=[3, 5])
    test('exp', lambda m: F.exp(m.a), a=[3, 5])
    test('pow_const', lambda m: m.a ** 4.2, a=[3, 5])
    test('pow', lambda m: m.a ** m.b, a=[3, 5], b=[7, 2])
    test('sigmoid', lambda m: F.sigmoid(m.a), a=[-4, 3, 5])
    test('relu', lambda m: F.relu(m.a), a=[-3, 3, 5])
    test('elu', lambda m: F.elu(m.a), a=[-3, 3, 5])
    test('reduce_sum', lambda m: F.sum(m.a, axis=0), a=[3, 5, 7])
    test('reduce_sum_neg_axis', lambda m: F.sum(m.a, axis=-1),
         a=aranges(2, 3, 5))
    test('reduce_sum_keepdims',
         lambda m: F.sum(m.a, axis=1, keepdims=True), a=aranges(2, 3, 5))
    test('reduce_sum_multi_axes',
         lambda m: F.sum(m.a, axis=(0, 2)), a=aranges(2, 3, 5))
    test('reduce_mean', lambda m: F.mean(m.a, axis=0), a=[3, 5, 7])
    test('reduce_mean_neg_axis', lambda m: F.mean(m.a, axis=-1),
         a=aranges(2, 3, 5))
    test('reduce_mean_keepdims',
         lambda m: F.mean(m.a, axis=1, keepdims=True), a=aranges(2, 3, 5))
    test('clip', lambda m: F.clip(m.a, -2.0, 4.0),
         a=[-3.0, -2.0, 3.0, 4.0, 5.0])
    test('reduce_mean_multi_axes',
         lambda m: F.mean(m.a, axis=(0, 2)), a=aranges(2, 3, 5))

    test('mulconst', lambda m: m.a * 1.23, a=[3, 5])
    test('mulbcast', lambda m: m.a * m.b, a=[[1, 2, 3], [4, 5, 6]], b=[7, 8, 9])

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
    test('matmul_ta_tb',
         lambda m: F.matmul(m.a, m.b, transa=True, transb=True),
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
    test('conv_transpose', lambda m: F.deconvolution_2d(m.a, m.b),
         a=aranges(1, 2, 7, 7),
         b=aranges(2, 5, 3, 3))
    # The 4th parameter is calculating gradients of bias more complex.
    test('conv_transpose_bias',
         lambda m: F.deconvolution_2d(m.a, m.b, b=m.c) * m.d,
         a=aranges(1, 2, 7, 7),
         b=aranges(2, 5, 3, 3),
         c=aranges(5),
         d=aranges(1, 5, 9, 9))
    test('grouped_conv', lambda m: F.convolution_2d(m.a, m.b, groups=3),
         a=aranges(1, 6, 11, 11),
         b=aranges(6, 2, 3, 3))
    # The 4th parameter is calculating gradients of bias more complex.
    test('grouped_conv_bias',
         lambda m: F.convolution_2d(m.a, m.b, b=m.c, groups=3) * m.d,
         a=aranges(1, 6, 11, 11),
         b=aranges(6, 2, 3, 3),
         c=aranges(6),
         d=aranges(1, 6, 9, 9))
    test('grouped_conv_transpose',
         lambda m: F.deconvolution_2d(m.a, m.b, groups=3),
         a=aranges(1, 6, 9, 9),
         b=aranges(6, 2, 3, 3))
    # The 4th parameter is calculating gradients of bias more complex.
    test('grouped_conv_transpose_bias',
         lambda m: F.deconvolution_2d(m.a, m.b, b=m.c, groups=3) * m.d,
         a=aranges(1, 6, 9, 9),
         b=aranges(6, 2, 3, 3),
         c=aranges(6),
         d=aranges(1, 6, 11, 11))
    test('grouped_conv_3d', lambda m: F.convolution_nd(m.a, m.b, groups=3),
         a=aranges(1, 6, 5, 5, 5),
         b=aranges(6, 2, 3, 3, 3))
    # NOTE: Enable this test after we support 4D convolution in GPU environment
    # test('grouped_conv_4d', lambda m: F.convolution_nd(m.a, m.b, groups=3),
    #      a=aranges(1, 6, 5, 5, 5, 5),
    #      b=aranges(6, 2, 3, 3, 3, 3))

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
    test('batch_normalization_2d',
         lambda m: F.batch_normalization(m.x, m.g, m.b) * m.r,
         x=aranges(2, 5),
         g=aranges(5),
         b=aranges(5),
         r=aranges(2, 5) % 7)
    test('fixed_batch_normalization',
         lambda m: F.fixed_batch_normalization(m.x, m.g, m.b, m.m, m.v) * m.r,
         x=aranges(2, 5, 3, 3) * 0.1,
         g=aranges(5),
         b=aranges(5),
         m=aranges(5),
         v=aranges(5),
         r=aranges(2, 5, 3, 3) % 7,
         test_params={'fixed_batch_norm': True})

    test('pad',
         lambda m: F.pad(m.x, 2, 'constant'),
         x=aranges(2, 5, 3, 3))

    # TODO(hamaji): Enable this test after fixing gradient of binary
    # ops with broadcast.
    # test('normalize',
    #      lambda m: F.normalize(m.x, axis=1),
    #      x=aranges(2, 5, 3, 3))

    # test case for computation_order
    test('tanh2', lambda m: F.tanh(F.tanh(m.a)), a=[0.3, 0.6])
    test('mul2', lambda m: (m.a * m.a) * m.a, a=[0.3, 0.6])
    test('max_pool2',
         lambda m: F.max_pooling_2d(
             F.max_pooling_2d(m.a, 2, stride=1, cover_all=False),
             2, stride=1, cover_all=False),
         a=aranges(2, 3, 5, 5) % 9)

    test('unpool',
         lambda m: F.unpooling_2d(m.a, 2, stride=2, cover_all=False),
         a=aranges(2, 3, 11, 11))

    test('concat_axis0',
         lambda m: F.concat((m.x, m.y), axis=0),
         x=aranges(2, 3, 2), y=aranges(3, 3, 2))
    test('concat_axis1',
         lambda m: F.concat((m.x, m.y), axis=1),
         x=aranges(2, 2, 2), y=aranges(2, 3, 2))

    def branched_conv(m):
        h = F.relu(m.a)
        return F.convolution_2d(h, m.b) + F.convolution_2d(h, m.c)

    test('branched_conv', branched_conv,
         a=aranges(1, 1, 5, 5),
         b=aranges(1, 1, 3, 3),
         c=aranges(1, 1, 3, 3),)

    test('split_2_axis_0',
         lambda m: sum(F.split_axis(m.x, 2, axis=0)),
         x=aranges(2, 3, 2))
    test('split_2_axis_1',
         lambda m: sum(F.split_axis(m.x, 3, axis=1)),
         x=aranges(2, 12, 2))

    def get_item(m):
        indices = aranges(5, 3).astype(np.int32) % 12
        return m.x[:, indices] * m.r
    test('get_item', get_item,
         x=aranges(1, 12, 32), r=aranges(1, 5, 3, 32))

    test('cast',
         lambda m: F.cast(m.x, np.float64),
         x=aranges(2, 12, 2))

    test('where',
         lambda m, cond: F.where(cond, m.x, m.y),
         x=aranges(20),
         y=-aranges(20),
         args=[np.array(np.random.randint(2, size=20), dtype=np.bool)])

    return tests


def main():
    for test in get_backprop_tests():
        test.generate()


if __name__ == '__main__':
    main()
