"""Yet another ONNX test generator for custom ops and new ops."""


import os

import chainer
import numpy as np
import onnx
from onnx import numpy_helper


F = chainer.functions


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def V(a):
    return chainer.variable.Variable(np.array(a))


def aranges(*shape):
    r = 1
    for d in shape:
        r *= d
    return np.arange(r).reshape(shape).astype(np.float32)


# From onnx/backend/test/case/node/__init__.py
def _extract_value_info(arr, name):
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape)


def expect(node, inputs, outputs, name):
    inputs = [v.array if hasattr(v, 'array') else v for v in inputs]
    outputs = [v.array if hasattr(v, 'array') else v for v in outputs]
    inputs = list(map(np.array, inputs))
    outputs = list(map(np.array, outputs))

    assert len(node.input) == len(inputs)
    assert len(node.output) == len(outputs)
    inputs_vi = [_extract_value_info(a, n)
                 for a, n in zip(inputs, node.input)]
    outputs_vi = [_extract_value_info(a, n)
                  for a, n in zip(outputs, node.output)]

    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    model = onnx.helper.make_model(graph, producer_name='backend-test')

    test_dir = os.path.join('out', name)
    test_data_set_dir = os.path.join(test_dir, 'test_data_set_0')
    makedirs(test_data_set_dir)
    with open(os.path.join(test_dir, 'model.onnx'), 'wb') as f:
        f.write(model.SerializeToString())
    for typ, values in [('input', zip(inputs, node.input)),
                        ('output', zip(outputs, node.output))]:
        for i, (value, name) in enumerate(values):
            filename = os.path.join(test_data_set_dir, '%s_%d.pb' % (typ, i))
            tensor = numpy_helper.from_array(value, name)
            with open(filename, 'wb') as f:
                f.write(tensor.SerializeToString())


def gen_select_item_test(test_name):
    input = V(aranges(4, 3))
    indices = V([1, 2, 0, 1])
    output = F.select_item(input, indices)

    node = onnx.helper.make_node(
        'OnikuxSelectItem',
        inputs=['input', 'indices'],
        outputs=['output'])
    expect(node, inputs=[input, indices], outputs=[output], name=test_name)


def gen_scan_sum_test(test_name):
    inputs1 = np.array([[4, 5, 6], [-4, -6, -5]])
    inputs2 = np.array([[1, 2, 3], [-3, -2, -1]])
    state = np.array([0, 0])
    out_state = []
    outputs = []
    for bi1, bi2, st in zip(inputs1, inputs2, state):
        outs = []
        for a, b in zip(bi1, bi2):
            ab = a - b
            r = ab + st
            outs.append(ab)
            st = r
        outputs.append(outs)
        out_state.append(st)
    outputs = np.array(outputs)

    inputs_vi = [_extract_value_info(inputs1[0][0], 'in%d' % i)
                 for i in range(1 + 2)]
    outputs_vi = [_extract_value_info(outputs[0][0], n)
                  for n in ['r', 'ab']]

    sub = onnx.helper.make_node('Sub', inputs=['a', 'b'], outputs=['ab'])
    add = onnx.helper.make_node('Add', inputs=['ab', 's'], outputs=['r'])
    body = onnx.helper.make_graph(
        nodes=[sub, add],
        name='body',
        inputs=inputs_vi,
        outputs=outputs_vi)

    node = onnx.helper.make_node(
        'Scan',
        body=body,
        num_scan_inputs=2,
        inputs=['state', 'inputs1', 'inputs2'],
        outputs=['out_state', 'outputs'])
    expect(node,
           inputs=[state, inputs1, inputs2],
           outputs=[out_state, outputs],
           name=test_name)


def gen_scan_sum_test(test_name):
    inputs1 = np.array([[4, 5, 6], [-4, -6, -5]])
    inputs2 = np.array([[1, 2, 3], [-3, -2, -1]])
    state = np.array([0, 0])
    out_state = []
    outputs = []
    for bi1, bi2, st in zip(inputs1, inputs2, state):
        outs = []
        for a, b in zip(bi1, bi2):
            ab = a - b
            r = ab + st
            outs.append(ab)
            st = r
        outputs.append(outs)
        out_state.append(st)
    outputs = np.array(outputs)

    inputs_vi = [_extract_value_info(inputs1[0][0], 'in%d' % i)
                 for i in range(1 + 2)]
    outputs_vi = [_extract_value_info(outputs[0][0], n)
                  for n in ['r', 'ab']]

    sub = onnx.helper.make_node('Sub', inputs=['a', 'b'], outputs=['ab'])
    add = onnx.helper.make_node('Add', inputs=['ab', 's'], outputs=['r'])
    body = onnx.helper.make_graph(
        nodes=[sub, add],
        name='body',
        inputs=inputs_vi,
        outputs=outputs_vi)

    node = onnx.helper.make_node(
        'Scan',
        body=body,
        num_scan_inputs=2,
        inputs=['state', 'inputs1', 'inputs2'],
        outputs=['out_state', 'outputs'])
    expect(node,
           inputs=[state, inputs1, inputs2],
           outputs=[out_state, outputs],
           name=test_name)


def make_constant_node(name, typ, value):
    tensor = onnx.helper.make_tensor(name + '_val', typ, (), value)
    node = onnx.helper.make_node('Constant', inputs=[], outputs=[name],
                                 value=tensor)
    return node


def gen_loop_simple_sum_test():
    def fn(test_name):
        input_state = np.array(0)
        state = input_state
        output = np.array(15)

        iter_vi = _extract_value_info(np.array(0), 'iter')
        cond_in_vi = _extract_value_info(np.array(True), 'cond_in')
        cond_vi = _extract_value_info(np.array(True), 'cond')
        inputs_vi = [_extract_value_info(state, 'in')]
        outputs_vi = [_extract_value_info(output, 'out')]

        sum = onnx.helper.make_node('Add', inputs=['in', 'iter'],
                                    outputs=['out'])
        loop_cnt = make_constant_node(
            'loop_cnt', onnx.TensorProto.INT64, [5])
        cond = onnx.helper.make_node('Less', inputs=['iter', 'loop_cnt'],
                                     outputs=['cond'])
        body = onnx.helper.make_graph(
            nodes=[sum, loop_cnt, cond],
            name='body',
            inputs=[iter_vi] + [cond_in_vi] + inputs_vi,
            outputs=[cond_vi] + outputs_vi)

        max_loop_cnt = make_constant_node(
            'max_loop_cnt', onnx.TensorProto.INT64, [7])
        first_cond = make_constant_node(
            'first_cond', onnx.TensorProto.BOOL, [True])
        node = onnx.helper.make_node(
            'Loop',
            body=body,
            inputs=['max_loop_cnt', 'first_cond', 'state'],
            outputs=['output'])
        expect(node,
               inputs=[np.array(7), np.array(True), state],
               outputs=[output],
               name=test_name)

    return fn


def gen_loop_sum_fact_test(test_name):
    # TODO(hamaji): Apparently, this test case is broken.
    inputs = (np.array([4, 5, 6]), np.array([1, 3, 2]))
    input_states = (np.array(0), np.array(1))
    states = input_states
    outputs = []
    for a, b in zip(*inputs):
        ab = a - b
        states = (states[0] + ab, states[1] * ab)
        outputs.append(ab)
    outputs = np.array(outputs)

    iter_vi = _extract_value_info(np.array(0), 'iter')
    cond_vi = _extract_value_info(np.array(True), 'cond')
    inputs_vi = [_extract_value_info(input, 'in%d' % i)
                 for i, input in enumerate(inputs)]
    outputs_vi = [_extract_value_info(outputs[0], n)
                  for n in ['sum', 'fact', 'ab']]

    sub = onnx.helper.make_node('Sub', inputs=['a', 'b'], outputs=['ab'])
    sum = onnx.helper.make_node('Add', inputs=['ab', 'in0'], outputs=['sum'])
    fact = onnx.helper.make_node('Mul', inputs=['ab', 'in1'], outputs=['fact'])
    three = onnx.helper.make_tensor("three", onnx.TensorProto.INT64, (), [3])
    loop_cnt = onnx.helper.make_node('Constant', inputs=[],
                                     outputs=['loop_cnt'], value=three)
    less = onnx.helper.make_node('Less', inputs=['iter', 'loop_cnt'],
                                 outputs=['less'])
    body = onnx.helper.make_graph(
        nodes=[sub, sum, fact, loop_cnt, less],
        name='body',
        inputs=[iter_vi] + [cond_vi] + inputs_vi,
        outputs=outputs_vi)

    node = onnx.helper.make_node(
        'Loop',
        body=body,
        inputs=['state1', 'state2', 'inputs1', 'inputs2'],
        outputs=['sum', 'fact', 'outputs'])
    expect(node,
           inputs=list(input_states) + list(inputs),
           outputs=list(states) + [outputs],
           name=test_name)


class TestCase(object):
    def __init__(self, name, func, fail=False):
        self.name = name
        self.func = func
        self.fail = fail


def get_tests():
    return [
        TestCase('extra_test_select_item', gen_select_item_test),
        TestCase('extra_test_loop_simple_sum', gen_loop_simple_sum_test()),
        TestCase('extra_test_loop_sum_fact', gen_loop_sum_fact_test, fail=True),
        TestCase('extra_test_scan_sum', gen_scan_sum_test, fail=True),
    ]


def main():
    for test in get_tests():
        test.func(test.name)
    # TODO(hamaji): Stop writing a file to scripts.
    with open('scripts/extra_test_stamp', 'w'): pass


if __name__ == '__main__':
    main()
