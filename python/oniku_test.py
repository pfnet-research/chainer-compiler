import os
import sys

oniku_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(oniku_root, 'build/python'))

import chainerx
import numpy as np
import oniku


def aranges(*shape):
    r = np.prod(shape)
    return np.arange(r).reshape(shape).astype(np.float32)


def test_inference():
    graph = oniku.load('out/ch2o_node_Linear/model.onnx')
    params = graph.params()
    input_names = graph.input_names()
    output_names = graph.output_names()
    assert len(input_names) == 1
    assert len(output_names) == 2

    xcvm = graph.compile()

    inputs = dict(params)
    t1 = chainerx.array(aranges(5, 7))
    inputs[input_names[0]] = t1

    y1 = chainerx.dot(t1, params['/l1/W'].T) + params['/l1/b']
    y2 = chainerx.dot(t1, params['/l2/W'].T)

    outputs = xcvm.run(inputs)
    assert len(outputs) == 2

    chainerx.testing.assert_allclose(y1, outputs[output_names[0]])
    chainerx.testing.assert_allclose(y2, outputs[output_names[1]])
