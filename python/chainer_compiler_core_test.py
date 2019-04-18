import os
import sys

import chainerx
import chainerx.testing
import numpy as np
import onnx

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'build/python'))
sys.path.append(os.path.join(project_root, 'python'))
sys.path.append(os.path.join(project_root, 'scripts'))

import chainer_compiler_core

import onnx_script


def aranges(*shape):
    r = np.prod(shape)
    return chainerx.array(np.arange(r).reshape(shape).astype(np.float32))


def test_inference():
    graph = chainer_compiler_core.load('out/ch2o_node_Linear/model.onnx')
    params = graph.params()
    input_names = graph.input_names()
    output_names = graph.output_names()
    assert len(input_names) == 1
    assert len(output_names) == 2

    xcvm = graph.compile()

    inputs = dict(params)
    t1 = aranges(5, 7)
    inputs[input_names[0]] = chainer_compiler_core.value(t1)

    y1 = chainerx.dot(t1, params['/l1/W'].array().T) + params['/l1/b'].array()
    y2 = chainerx.dot(t1, params['/l2/W'].array().T)

    outputs = xcvm.run(inputs)
    assert len(outputs) == 2

    chainerx.testing.assert_allclose(y1, outputs[output_names[0]].array())
    chainerx.testing.assert_allclose(y2, outputs[output_names[1]].array())

    assert 'op_type: "ChainerLinear"' in graph.dump()


def test_backprop():
    graph = chainer_compiler_core.load('out/ch2o_node_Linear_backprop/model.onnx')
    params = graph.params()
    input_names = graph.input_names()
    output_names = graph.output_names()
    assert len(input_names) == 1
    assert len(output_names) == 1

    fwd_graph, bwd_graph = graph.backward()
    assert len(fwd_graph.input_names()) == 1
    assert len(fwd_graph.output_names()) == 3
    assert len(bwd_graph.input_names()) == 3
    assert len(bwd_graph.output_names()) == 2

    fwd = fwd_graph.compile()
    bwd = bwd_graph.compile()

    fwd_inputs = dict(params)
    t1 = aranges(5, 7)
    fwd_inputs[input_names[0]] = chainer_compiler_core.value(t1)

    loss = (chainerx.dot(t1, params['/l1/W'].array().T) +
            params['/l1/b'].array())

    fwd_outputs = fwd.run(fwd_inputs)
    assert len(fwd_outputs) == 3

    chainerx.testing.assert_allclose(
        loss, fwd_outputs[output_names[0]].array())

    grad_loss = aranges(*loss.shape) + 4.2

    bwd_inputs = {}
    for name in fwd_graph.output_names():
        iname = name
        value = fwd_outputs[name]
        if name in output_names:
            iname = 'grad_in@' + name
            value = chainer_compiler_core.value(grad_loss)
        bwd_inputs[iname] = value

    bwd_outputs = bwd.run(bwd_inputs)

    grad_w = chainerx.dot(grad_loss.T, t1)
    chainerx.testing.assert_allclose(
        grad_w, bwd_outputs['grad_out@/l1/W'].array())
    grad_b = chainerx.sum(grad_loss, axis=0)
    chainerx.testing.assert_allclose(
        grad_b, bwd_outputs['grad_out@/l1/b'].array())


def test_custom_op():
    gb = onnx_script.GraphBuilder('pytest_custom_op')
    a = np.array(13)
    b = np.array(4)
    c = np.array(10)
    a_v = gb.input('a', a)
    b_v = gb.input('b', b)
    c_v = gb.input('c', c)

    def custom_func(a, b, c):
        return a - b, a * b - c

    y, z = custom_func(a, b, c)
    y_v = 'y'
    z_v = 'z'

    gb.ChainerDoSomething([a_v, b_v, c_v],
                          outputs=[y_v, z_v],
                          function_name='CustomFunction')
    gb.output(y_v, y)
    gb.output(z_v, z)

    gb.gen_test()

    graph = chainer_compiler_core.load('out/pytest_custom_op/model.onnx')
    params = graph.params()
    input_names = graph.input_names()
    output_names = graph.output_names()
    assert len(input_names) == 3
    assert len(output_names) == 2

    xcvm = graph.compile()

    inputs = {}
    for n, v in [('a', a), ('b', b), ('c', c)]:
        inputs[n] = chainer_compiler_core.value(chainerx.array(v))

    outputs = xcvm.run(inputs)
    assert len(outputs) == 2
