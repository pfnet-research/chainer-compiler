#!/usr/bin/python3

import os

import chainer
import numpy as np
from onnx import onnx_pb
import onnx_chainer


def replace_id(builtins=__builtins__):
    orig_id = id

    def my_id(x):
        if isinstance(x, chainer.Parameter):
            return x.name
        return orig_id(x)
    builtins.id = my_id


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


class AnyModel(chainer.Chain):
    def __init__(self, fn, params):
        super(AnyModel, self).__init__()
        with self.init_scope():
            for name, value in params.items():
                setattr(self, name, chainer.Parameter(value, name=name))
        self.fn = fn

    def __call__(self, *args):
        result = self.fn(self, *args)
        return result


def create_backprop_test(test_name, fn, expected_extra_params=None, **kwargs):
    test_dir = 'out/backprop_test_%s' % test_name
    test_model_path = os.path.join(test_dir, 'model.onnx')
    test_data_dir = os.path.join(test_dir, 'test_data_set_0')
    makedirs(test_data_dir)

    params = {}
    for name, value in kwargs.items():
        params[name] = np.array(value, np.float32)
    model = AnyModel(fn, params)

    onnx_chainer.export(model, (),
                        filename=test_model_path,
                        graph_name='backprop_test_' + test_name)

    model.cleargrads()
    result = model()
    result.grad = np.ones(result.shape, result.dtype)
    result.backward()

    param_names = []
    output_names = []
    xmodel = onnx_pb.ModelProto()
    with open(test_model_path, 'rb') as f:
        xmodel.ParseFromString(f.read())
        for param in xmodel.graph.initializer:
            assert param.name
            param_names.append(param.name)
        for output in xmodel.graph.output:
            assert output.name
            output_names.append(output.name)

    assert len(output_names) == 1
    expected_param_names = list(params.keys())
    if expected_extra_params is not None:
        expected_param_names += expected_extra_params
    if sorted(expected_param_names) != sorted(param_names):
        print('expected=%s actual=%s' %
              (list(sorted(expected_param_names)), list(sorted(param_names))))
        assert False

    outputs = [(output_names[0], result)]
    for name in sorted(params):
        value = getattr(model, name).grad
        outputs.append(('grad_out@' + name, value))

    for i, (xname, value) in enumerate(outputs):
        tensor = onnx_pb.TensorProto(name=xname,
                                     dims=value.shape,
                                     data_type=onnx_pb.TensorProto.FLOAT)
        tensor.float_data.extend(np.array(value.data).flat)
        with open(os.path.join(test_data_dir, 'output_%d.pb' % i), 'wb') as f:
            f.write(tensor.SerializeToString())


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

    def aranges(shape):
        r = 1
        for d in shape:
            r *= d
        return np.arange(r).reshape(shape)

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

    # TODO(hamaji): To get this work, the following TODO in
    # onnx-chainer should be fixed:
    # https://github.com/chainer/onnx-chainer/blob/master/onnx_chainer/functions/array.py#L79
    # test('reshape', lambda m: F.reshape(m.a, (1, 2, 1)),
    #      expected_extra_params=['None'], a=[3, 5])

    test('sqrt', lambda m: F.sqrt(m.a), a=[3, 5])

    # ONNX chainer creates an extra parameter named 'None' for bias of
    # Gemm.
    test('matmul', lambda m: F.matmul(m.a, m.b),
         expected_extra_params=['None'],
         a=[[3, 5], [7, 4], [2, 6]], b=[[2, 4, 8, 9], [4, 2, 12, 6]])
    test('matmul_ta', lambda m: F.matmul(m.a, m.b, transa=True),
         expected_extra_params=['None'],
         a=np.transpose([[3, 5], [7, 4], [2, 6]]),
         b=[[2, 4, 8, 9], [4, 2, 12, 6]])
    test('matmul_tb', lambda m: F.matmul(m.a, m.b, transb=True),
         expected_extra_params=['None'],
         a=[[3, 5], [7, 4], [2, 6]],
         b=np.transpose([[2, 4, 8, 9], [4, 2, 12, 6]]))
    test('matmul_ta_tb', lambda m: F.matmul(m.a, m.b, transa=True, transb=True),
         expected_extra_params=['None'],
         a=np.transpose([[3, 5], [7, 4], [2, 6]]),
         b=np.transpose([[2, 4, 8, 9], [4, 2, 12, 6]]))
    test('gemm', lambda m: F.linear(m.a, m.b, b=m.c),
         a=[[3, 5], [7, 4], [2, 6]],
         b=np.transpose([[2, 4, 8, 9], [4, 2, 12, 6]]),
         c=[4, 5, 6, 7])

    test('conv', lambda m: F.convolution_2d(m.a, m.b),
         a=aranges((1, 1, 5, 5)),
         b=aranges((1, 1, 3, 3)))
    # The 4th parameter is calculating gradients of bias more complex.
    test('conv_bias', lambda m: F.convolution_2d(m.a, m.b, b=m.c) * m.d,
         a=aranges((1, 2, 5, 5)),
         b=aranges((4, 2, 3, 3)),
         c=aranges((4,)),
         d=aranges((1, 4, 3, 3)))
    test('max_pool', lambda m: F.max_pooling_2d(m.a, 3, stride=1,
                                                cover_all=False) * m.b,
         a=aranges((2, 3, 5, 5)) % 9,
         b=aranges((2, 3, 3, 3)))
    test('average_pool', lambda m: F.average_pooling_2d(m.a, 3,
                                                        stride=1) * m.b,
         a=aranges((2, 3, 5, 5)) % 9,
         b=aranges((2, 3, 3, 3)))

    test('log_softmax', lambda m: F.log_softmax(m.a),
         a=[[2, 4, 8], [3, 1, 9], [4, 12, 6]])
    # Multiply `b` to avoid nearly zero gradients.
    test('softmax_axis0', lambda m: F.softmax(m.a, axis=0) * m.b,
         a=[[2, 4, 8], [3, 1, 9], [4, 12, 6]],
         b=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    test('softmax_axis1', lambda m: F.softmax(m.a, axis=1) * m.b,
         a=[[2, 4, 8], [3, 1, 9], [4, 12, 6]],
         b=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    return tests


def main():
    replace_id()
    for test in get_backprop_tests():
        test.generate()


if __name__ == '__main__':
    main()
