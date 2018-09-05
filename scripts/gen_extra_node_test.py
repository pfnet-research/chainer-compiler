"""Yet another ONNX test generator for custom ops and new ops."""


import os
import shutil

import chainer
import numpy as np
import onnx
from onnx import numpy_helper


F = chainer.functions


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
    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in node.output if (x != '')]
    inputs = [v.array if hasattr(v, 'array') else v for v in inputs]
    outputs = [v.array if hasattr(v, 'array') else v for v in outputs]
    inputs = list(map(np.array, inputs))
    outputs = list(map(np.array, outputs))

    assert len(present_inputs) == len(inputs)
    assert len(present_outputs) == len(outputs)
    inputs = list(zip(present_inputs, inputs))
    outputs = list(zip(present_outputs, outputs))
    inputs_vi = [_extract_value_info(a, n) for n, a in inputs]
    outputs_vi = [_extract_value_info(a, n) for n, a in outputs]

    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    gen_test(graph, inputs, outputs, name)


def gen_test(graph, inputs, outputs, name):
    model = onnx.helper.make_model(graph, producer_name='backend-test')

    test_dir = os.path.join('out', name)
    test_data_set_dir = os.path.join(test_dir, 'test_data_set_0')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_data_set_dir)
    with open(os.path.join(test_dir, 'model.onnx'), 'wb') as f:
        f.write(model.SerializeToString())
    for typ, values in [('input', inputs), ('output', outputs)]:
        for i, (name, value) in enumerate(values):
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


def gen_loop_test(max_trip_count=7,
                  cond_trip_count=6,
                  terminal_condition=True,
                  has_scan_outputs=False):
    def fn(test_name):
        input_state = np.array(0)
        state = input_state

        trip_counts = []
        if max_trip_count is not None:
            trip_counts.append(max_trip_count)
        if terminal_condition is False:
            trip_counts.append(0)
        elif terminal_condition is not None:
            trip_counts.append(cond_trip_count)
        trip_count = min(trip_counts)
        output = np.array(sum(range(trip_count)))
        scan_outputs = []
        if has_scan_outputs:
            scan_outputs = [
                np.array(list(sum(range(i + 1)) for i in range(trip_count))),
                np.array(list(i * i for i in range(trip_count))),
                np.array(list(range(trip_count)))]

        iter_vi = _extract_value_info(np.array(0), 'iter')
        cond_in_vi = _extract_value_info(np.array(True), 'cond_in')
        cond_vi = _extract_value_info(np.array(True), 'cond')
        inputs_vi = [_extract_value_info(state, 'in')]
        outputs_vi = [_extract_value_info(output, 'out')]
        square = []
        if has_scan_outputs:
            outputs_vi.append(_extract_value_info(output, 'out'))
            outputs_vi.append(_extract_value_info(output, 'square'))
            outputs_vi.append(_extract_value_info(output, 'iter'))
            square = [onnx.helper.make_node('Mul', inputs=['iter', 'iter'],
                                            outputs=['square'])]

        body_out = onnx.helper.make_node('Add', inputs=['in', 'iter'],
                                         outputs=['out'])
        loop_cnt = make_constant_node(
            'loop_cnt', onnx.TensorProto.INT64, [cond_trip_count - 1])
        cond = onnx.helper.make_node('Less', inputs=['iter', 'loop_cnt'],
                                     outputs=['cond'])
        body = onnx.helper.make_graph(
            nodes=[body_out, loop_cnt, cond] + square,
            name='body',
            inputs=[iter_vi] + [cond_in_vi] + inputs_vi,
            outputs=[cond_vi] + outputs_vi)

        max_trip_cnt_sym = ''
        max_trip_cnt_value = []
        if max_trip_count is not None:
            max_trip_cnt_sym = 'max_trip_cnt'
            max_trip_cnt_value = [np.array(max_trip_count)]

        terminal_cond_sym = ''
        terminal_cond_value = []
        if terminal_condition is not None:
            terminal_cond_sym = 'terminal_cond'
            terminal_cond_value = [np.array(terminal_condition)]

        output_syms = ['output']
        output_values = [output]
        if has_scan_outputs:
            output_syms += ['history', 'square', 'range']
            output_values += scan_outputs

        node = onnx.helper.make_node(
            'Loop',
            body=body,
            inputs=[max_trip_cnt_sym, terminal_cond_sym, 'state'],
            outputs=output_syms)
        expect(node,
               inputs=max_trip_cnt_value + terminal_cond_value + [state],
               outputs=output_values,
               name=test_name)

    return fn


def gen_sequence_test(test_name):
    inputs = [np.array(a) for a in [[1, 2], [3, 4], [5, 6]]]
    nodes = []
    nodes.append(onnx.helper.make_node(
        'OnikuxSequenceCreate',
        inputs=[],
        outputs=['seq0']))

    for i, input in enumerate(inputs):
        nodes.append(onnx.helper.make_node(
            'OnikuxSequenceAppend',
            inputs=['seq%d' % i, 'in%d' % i],
            outputs=['seq%d' % (i + 1)]))

    index_value = 1
    nodes.append(make_constant_node(
        'index', onnx.TensorProto.INT64, [index_value]))
    nodes.append(onnx.helper.make_node(
        'OnikuxSequenceLookup',
        inputs=['seq3', 'index'],
        outputs=['lookup_result']))
    nodes.append(onnx.helper.make_node(
        'OnikuxSequenceStack',
        inputs=['seq3'],
        outputs=['stack3_result']))
    nodes.append(onnx.helper.make_node(
        'OnikuxSequenceStack',
        inputs=['seq2'],
        outputs=['stack2_result']))

    outputs = [
        ('lookup_result', np.array([3, 4])),
        ('stack3_result', np.stack(inputs)),
        ('stack2_result', np.stack(inputs[0:2])),
    ]
    inputs = [('in%d' % i, input) for i, input in enumerate(inputs)]
    inputs_vi = [_extract_value_info(a, n) for n, a in inputs]
    outputs_vi = [_extract_value_info(a, n) for n, a in outputs]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name=test_name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    gen_test(graph, inputs, outputs, name=test_name)


class TestCase(object):
    def __init__(self, name, func, fail=False):
        self.name = name
        self.func = func
        self.fail = fail


def get_tests():
    return [
        TestCase('extra_test_select_item', gen_select_item_test),
        TestCase('extra_test_loop_basic', gen_loop_test()),
        TestCase('extra_test_loop_max_trip_count',
                 gen_loop_test(max_trip_count=4)),
        TestCase('extra_test_loop_no_max_trip_count',
                 gen_loop_test(max_trip_count=None)),
        TestCase('extra_test_loop_false_cond',
                 gen_loop_test(terminal_condition=False)),
        TestCase('extra_test_loop_no_cond',
                 gen_loop_test(terminal_condition=None)),
        TestCase('extra_test_loop_scan_out',
                 gen_loop_test(has_scan_outputs=True)),

        TestCase('extra_test_scan_sum', gen_scan_sum_test, fail=True),

        TestCase('extra_test_sequence', gen_sequence_test),
    ]


def main():
    for test in get_tests():
        test.func(test.name)
    # TODO(hamaji): Stop writing a file to scripts.
    with open('scripts/extra_test_stamp', 'w'): pass


if __name__ == '__main__':
    main()
