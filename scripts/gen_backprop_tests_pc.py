#!/usr/bin/python3

import os
import sys

import chainer
import numpy as np
import onnx
from onnx import onnx_pb

my_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(my_path))
sys.path.append(os.path.join(my_path, 'ch2o'))
from ch2o import compiler
from ch2o import test_args
from ch2o import testcasegen
from oniku.scripts import onnx_chainer_util

F = chainer.functions
L = chainer.links


def create_backprop_test(test_name, model, input_values):
    test_dir = 'out/backprop_test_pc_%s' % test_name
    test_data_set_dir = os.path.join(test_dir, 'test_data_set_0')
    onnx_chainer_util.makedirs(test_data_set_dir)

    xmodel = compiler(model)
    all_input_tensors = xmodel.graph.input
    output_tensors = xmodel.graph.output

    chainer.config.train = True
    model.cleargrads()
    output_values = model(*map(chainer.variable.Variable, input_values))
    if not isinstance(output_values, (list, tuple)):
        output_values = (output_values,)
    for output_value in output_values:
        output_value.grad = np.ones(output_value.shape, output_value.dtype)
        output_value.backward()

    testcasegen.edit_onnx_protobuf(xmodel, model)

    with open(os.path.join(test_dir, 'model.onnx'), 'wb') as fp:
        fp.write(xmodel.SerializeToString())

    initializer_names = set()
    for initializer in xmodel.graph.initializer:
        initializer_names.add(initializer.name)
    input_tensors = []
    for input_tensor in all_input_tensors:
        if input_tensor.name not in initializer_names:
            input_tensors.append(input_tensor)

    assert len(input_tensors) == len(input_values)
    assert len(output_tensors) == len(output_values)

    outputs = []
    for tensor, value in zip(output_tensors, output_values):
        outputs.append((tensor, value.array))
    for name, param in sorted(model.namedparams()):
        bp_name = onnx.helper.make_tensor_value_info(
            'grad_out@' + name, onnx.TensorProto.FLOAT, ())
        outputs.append((bp_name, param.grad))

    testcasegen.dump_test_inputs_outputs(
        list(zip(input_tensors, input_values)),
        outputs,
        test_data_set_dir)


class BackpropTest(object):
    def __init__(self, name, model, inputs, rtol=1e-4):
        self.name = name
        self.model = model
        self.inputs = inputs
        self.rtol = rtol

    def generate(self):
        create_backprop_test(self.name, self.model, self.inputs)


def get_backprop_tests():
    F = chainer.functions
    tests = []

    def test(name, model, *inputs, rtol=1e-4):
        tests.append(BackpropTest(name, model, inputs, rtol))

    def aranges(*shape):
        r = 1
        for d in shape:
            r *= d
        return np.arange(r).reshape(shape).astype(np.float32)

    class Nop(chainer.Chain):
        def forward(self, x):
            return x

    test('nop', Nop(), aranges(2, 3))

    class AddSelf(chainer.Chain):
        def forward(self, x):
            return x + x

    test('add_self', AddSelf(), aranges(2, 3))

    class Linear(chainer.Chain):
        def __init__(self):
            super(Linear, self).__init__()
            with self.init_scope():
                self.l1 = L.Linear(None, 10)

        def forward(self, x):
            return F.relu(self.l1(x))

    test('linear', Linear(), aranges(2, 3))

    class SoftmaxCrossEntropy(chainer.Chain):
        def __init__(self):
            super(SoftmaxCrossEntropy, self).__init__()
            with self.init_scope():
                self.l1 = L.Linear(None, 10)

        def forward(self, x, t):
            return F.softmax_cross_entropy(self.l1(x), t)

    test('softmax_cross_entropy', SoftmaxCrossEntropy(),
         aranges(2, 3), np.array([1, 0]))

    class LRN(chainer.Chain):
        def __init__(self):
            super(LRN, self).__init__()
            with self.init_scope():
                self.l1 = L.Linear(None, 10)

        def forward(self, x):
            return F.local_response_normalization(self.l1(x))

    test('lrn', LRN(), aranges(2, 3))

    return tests


def main():
    for test in get_backprop_tests():
        np.random.seed(42)
        test.generate()


if __name__ == '__main__':
    sys.argv.append('--quiet')
    sys.argv.append('/tmp/dummy_dir')
    main()
