import os
import sys

import chainerx
import chainerx.testing
import numpy as np


oniku_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(oniku_root, 'build/python'))
sys.path.append(os.path.join(oniku_root, 'python'))

import oniku_core


def test_dropout_inference():
    graph = oniku_core.load(
        'onnx/onnx/backend/test/data/node/test_dropout_random/model.onnx')
    input_names = graph.input_names()
    output_names = graph.output_names()
    assert len(input_names) == 1
    assert len(output_names) == 1

    xcvm = graph.compile()
    input = chainerx.array(np.random.normal(size=(3, 4, 5)))
    inputs = {input_names[0]: oniku_core.value(input)}
    outputs = xcvm.run(inputs)
    output = outputs[output_names[0]].array()

    chainerx.testing.assert_allclose(input, output)
