#!/usr/bin/python3

import os
import sys

import chainer
import numpy as np
from onnx import onnx_pb

my_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(my_path))
sys.path.append(os.path.join(my_path, 'pc'))
from oniku.pc import chainer2onnx
from oniku.pc import test_args
from oniku.pc import testcasegen
from oniku.scripts import onnx_chainer_util

L = chainer.links


def create_backprop_test(test_name, model, input_values):
    test_dir = 'out/backprop_test_pc_%s' % test_name
    test_data_set_dir = os.path.join(test_dir, 'test_data_set_0')
    onnx_chainer_util.makedirs(test_data_set_dir)

    xmodel, input_tensors, output_tensors = chainer2onnx.chainer2onnx(
        model, model.forward)

    chainer.config.train = False
    output_values = testcasegen.run_chainer_model(
        model, tuple(map(chainer.variable.Variable, input_values)), 'loss')
    assert len(input_tensors) == len(input_values)
    assert len(output_tensors) == len(output_values)

    with open(os.path.join(test_dir, 'model.onnx'), 'wb') as fp:
        fp.write(xmodel.SerializeToString())

    initializer_names = set()
    for initializer in xmodel.graph.initializer:
        initializer_names.add(initializer.name)
    input_names = []
    for input_tensor in input_tensors:
        if input_tensor.name not in initializer_names:
            input_names.append(input_tensor.name)

    outputs = []
    for tensor, value in zip(output_tensors, output_values):
        outputs.append((tensor.name, value))

    testcasegen.dump_test_inputs_outputs(
        list(zip(input_names, input_values)),
        outputs,
        test_data_set_dir)


class BackpropTest(object):
    def __init__(self, name, model, inputs):
        self.name = name
        self.model = model
        self.inputs = inputs

    def generate(self):
        create_backprop_test(self.name, self.model, self.inputs)


def get_backprop_tests():
    F = chainer.functions
    tests = []

    def test(name, model, *inputs):
        tests.append(BackpropTest(name, model, inputs))

    def aranges(*shape):
        r = 1
        for d in shape:
            r *= d
        return np.arange(r).reshape(shape).astype(np.float32)

    class Nop(chainer.Chain):
        def forward(self, x):
            return x

    test('nop', Nop(), aranges(2, 3))

    return tests


def main():
    for test in get_backprop_tests():
        test.generate()
    # TODO(hamaji): Stop writing a file to scripts.
    with open('scripts/backprop_test_pc_stamp', 'w'): pass


if __name__ == '__main__':
    sys.argv.append('--quiet')
    main()
