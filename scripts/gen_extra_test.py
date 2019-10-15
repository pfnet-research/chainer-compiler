"""Yet another ONNX test generator for custom ops and new ops."""


import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import onnx

import onnx_script
import test_case

import gen_chainercv_op_tests
import sentiment


_extract_value_info = onnx_script._extract_value_info
make_constant_node = onnx_script.make_constant_node
gen_test = onnx_script.gen_test
Seq = onnx_script.Seq


def V(a):
    return chainer.variable.Variable(np.array(a))


def aranges(*shape):
    r = np.prod(shape)
    return np.arange(r).reshape(shape).astype(np.float32)


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


def gen_negative_reshape_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    v = np.array([2, 3, 4])
    v_v = gb.const(v)
    shape_v = gb.const([-1, 3])
    reshaped_v = gb.Reshape([v_v, shape_v])
    gb.output(reshaped_v, v.reshape((-1, 3)))
    gb.gen_test()


def gen_inf_nan_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    one_v = gb.const(1.0)
    none_v = gb.const(-1.0)
    zero_v = gb.const(0.0)
    inf_v = gb.Div([one_v, zero_v])
    ninf_v = gb.Div([none_v, zero_v])
    nan_v = gb.Log([none_v])
    gb.output(inf_v, np.inf)
    gb.output(ninf_v, -np.inf)
    gb.output(nan_v, -np.nan)
    gb.gen_test()


def gen_select_item_test(test_name):
    input = V(aranges(4, 3))
    indices = V([1, 2, 0, 1])
    output = F.select_item(input, indices)

    node = onnx.helper.make_node(
        'ChainerSelectItem',
        inputs=['input', 'indices'],
        outputs=['output'])
    expect(node, inputs=[input, indices], outputs=[output], name=test_name)


def gen_scan_sum_test(test_name):
    # TODO(hamaji): Rewrite with onnx_script.
    inputs1 = np.array([[4, 5, 6], [-4, -6, -5]])
    inputs2 = np.array([[1, 2, 3], [-3, -2, -1]])
    state = np.array(0)
    out_state = []
    outputs = []
    out_all_states = []
    for bi1, bi2 in zip(inputs1, inputs2):
        st = state
        outs = []
        all_states = []
        for a, b in zip(bi1, bi2):
            ab = a - b
            r = ab + st
            outs.append(ab)
            all_states.append(st)
            st = r
        outputs.append(outs)
        out_state.append(st)
        out_all_states.append(all_states)
    outputs = np.array(outputs)

    inputs_vi = [_extract_value_info(inputs1[0][0], n)
                 for n in ['s', 'a', 'b']]
    outputs_vi = [_extract_value_info(outputs[0][0], n)
                  for n in ['r', 'ab', 'so']]

    sub = onnx.helper.make_node('Sub', inputs=['a', 'b'], outputs=['ab'])
    add = onnx.helper.make_node('Add', inputs=['ab', 's'], outputs=['r'])
    ident = onnx.helper.make_node('Identity', inputs=['s'], outputs=['so'])
    body = onnx.helper.make_graph(
        nodes=[sub, add, ident],
        name='body',
        inputs=inputs_vi,
        outputs=outputs_vi)

    node = onnx.helper.make_node(
        'Scan',
        body=body,
        num_scan_inputs=2,
        inputs=['state', 'inputs1', 'inputs2'],
        outputs=['out_state', 'outputs', 'out_all_states'])
    expect(node,
           inputs=[state, inputs1, inputs2],
           outputs=[out_state, outputs, out_all_states],
           name=test_name)


def gen_if_test(cond):
    def fn(test_name):
        tb = onnx_script.GraphBuilder(test_name + '_true')
        for i in [42, 99]:
            true_value_v = tb.const(i)
            tb.output(true_value_v, i)

        fb = onnx_script.GraphBuilder(test_name + '_false')
        for i in [-42, -99]:
            false_value_v = fb.const(i)
            fb.output(false_value_v, i)

        gb = onnx_script.GraphBuilder(test_name)
        cond_v = gb.input('cond', cond)
        out1_v, out2_v = gb.If([cond_v],
                               then_branch=tb.make_graph(),
                               else_branch=fb.make_graph(),
                               outputs=['42', '99'])
        gb.output(out1_v, 42 if cond else -42)
        gb.output(out2_v, 99 if cond else -99)
        gb.gen_test()

    return fn


def gen_if_with_input_test(cond):
    def fn(test_name):
        tb = onnx_script.GraphBuilder(test_name + '_true')
        input_v = tb.input('input', 42)
        tb.output(tb.Identity([input_v]), 42)

        fb = onnx_script.GraphBuilder(test_name + '_false')
        input_v = fb.input('input', 42)
        fb.output(fb.Neg([input_v]), 42)

        gb = onnx_script.GraphBuilder(test_name)
        cond_v = gb.input('cond', cond)
        in_v = gb.input('in', 42)
        out_v = gb.If([cond_v, in_v],
                      then_branch=tb.make_graph(),
                      else_branch=fb.make_graph())
        gb.output(out_v, 42 if cond else -42)
        gb.gen_test()

    return fn


def gen_if_with_external_test(cond):
    def fn(test_name):
        gb = onnx_script.GraphBuilder(test_name)
        in0_v = gb.input('in0', 42)
        in1_v = gb.input('in1', 99)
        in2_v = gb.input('in2', 100)

        tb = onnx_script.GraphBuilder(test_name + '_true')
        tb.output(tb.Add([in0_v, in1_v]), 42)

        fb = onnx_script.GraphBuilder(test_name + '_false')
        fb.output(fb.Sub([in1_v, in2_v]), 42)

        cond_v = gb.input('cond', cond)
        out_v = gb.If([cond_v],
                      then_branch=tb.make_graph(),
                      else_branch=fb.make_graph())
        gb.output(out_v, 42 + 99 if cond else 99 - 100)
        gb.gen_test()

    return fn


def gen_loop_test(max_trip_count=7,
                  cond_trip_count=6,
                  terminal_condition=True,
                  has_scan_outputs=False):
    # TODO(hamaji): Rewrite with onnx_script.
    def fn(test_name):
        input_state = np.array(42)
        state = input_state

        trip_counts = []
        if max_trip_count is not None:
            trip_counts.append(max_trip_count)
        if terminal_condition is False:
            trip_counts.append(0)
        elif terminal_condition is not None:
            # `cond_trip_count` is not checked until the first
            # iteration finishes.
            trip_counts.append(max(cond_trip_count, 1))
        trip_count = min(trip_counts)

        output = np.array(sum(range(trip_count)) + 11 * trip_count + 42)
        scan_outputs = []
        if has_scan_outputs:
            scan_outputs = [
                np.array([sum(range(i + 1)) + 11 * (i + 1) + 42
                          for i in range(trip_count)]),
                np.array([i * i for i in range(trip_count)]),
                np.array(list(range(trip_count)))]

        iter_vi = _extract_value_info(np.array(0), 'iter')
        cond_in_vi = _extract_value_info(np.array(True), 'cond_in')
        cond_vi = _extract_value_info(np.array(True), 'cond')
        inputs_vi = [_extract_value_info(state, 'in')]
        outputs_vi = [_extract_value_info(output, 'out')]
        nodes = []
        if has_scan_outputs:
            outputs_vi.append(_extract_value_info(output, 'out2'))
            outputs_vi.append(_extract_value_info(output, 'square'))
            outputs_vi.append(_extract_value_info(output, 'iter2'))
            nodes.append(onnx.helper.make_node('Identity', inputs=['out'],
                                               outputs=['out2']))
            nodes.append(onnx.helper.make_node('Identity', inputs=['iter'],
                                               outputs=['iter2']))
            nodes.append(onnx.helper.make_node('Mul', inputs=['iter', 'iter'],
                                               outputs=['square']))

        nodes.append(make_constant_node(
            'const_11', onnx.TensorProto.INT64, 11))
        nodes.append(onnx.helper.make_node('Sum',
                                           inputs=['in', 'iter', 'const_11'],
                                           outputs=['out']))
        nodes.append(make_constant_node(
            'loop_cnt', onnx.TensorProto.INT64, cond_trip_count - 1))
        nodes.append(onnx.helper.make_node('Less', inputs=['iter', 'loop_cnt'],
                                           outputs=['cond']))
        body = onnx.helper.make_graph(
            nodes=nodes,
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


def gen_loop_use_enclosing_test():
    def fn(test_name):
        gb = onnx_script.GraphBuilder(test_name)
        init = np.array(10, np.float32)
        init_v = gb.param('init', init)
        external = np.array(42, np.float32)
        external_v = gb.param('external', external)

        bb = onnx_script.GraphBuilder(test_name + '_body')
        iter_v = bb.input('iter', np.array(0))
        cond_v = bb.input('cond', np.array(True))

        state_v = bb.input('state', init)
        result_v = bb.Add([state_v, external_v])
        cond_v = bb.const(True)
        bb.output(cond_v, np.array(True))
        bb.output(result_v, init)

        num_iter_v = gb.const(5)
        true_v = gb.const(True)
        out_v = gb.Loop([num_iter_v, true_v, init_v],
                        body=bb.make_graph())

        expected = float(5 * 42 + 10)
        gb.output(out_v, expected)

        gb.gen_test()

    return fn


def gen_backprop_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    i = np.array(42, np.float32)
    j = np.array(99, np.float32)

    i_v = gb.param('i', i)
    j_v = gb.param('j', j)

    r_v = gb.Mul([i_v, j_v])

    gb.output(r_v, i * j)
    gb.gradient(i_v, j)
    gb.gradient(j_v, i)
    gb.gen_test()


def gen_concat_backprop_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    i = np.array([42], np.float32)
    j = np.array([99], np.float32)

    i_v = gb.param('i', i)
    j_v = gb.param('j', j)

    concat_v = gb.Concat([i_v, j_v], axis=0)
    m = np.array([2, 3], np.float32)
    r_v = gb.Mul([concat_v, gb.const(m)])
    r = np.concatenate([i, j]) * m

    gb.output(r_v, r)
    gb.gradient(i_v, np.array([2], np.float32))
    gb.gradient(j_v, np.array([3], np.float32))
    gb.gen_test()


# Borrowed from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/framework/while_gradients_test.cc
def gen_loop_backprop_test(ii, ji, ki, gi, gj, gk):
    i, j, k = ii, ji, ki
    while i < 10:
        i += j
        j += 1
    expected = np.array(i + j + k, np.float32)

    def fn(test_name):
        gb = onnx_script.GraphBuilder(test_name)
        i = np.array(ii, np.float32)
        j = np.array(ji, np.float32)
        k = np.array(ki, np.float32)
        i_v = gb.param('i', i)
        j_v = gb.param('j', j)
        k_v = gb.param('k', k)

        bb = onnx_script.GraphBuilder(test_name + '_body')
        iter_v = bb.input('iter', np.array(0))
        cond_v = bb.input('cond', np.array(True))
        bi_v = bb.input('bi', i)
        bj_v = bb.input('bj', j)
        bk_v = bb.input('bk', k)
        one_v = bb.const(1.0)
        ni_v = bb.Add([bi_v, bj_v])
        nj_v = bb.Add([bj_v, one_v])
        nk_v = bb.Identity([bk_v])
        ten_v = bb.const(10.0)
        cond_v = bb.Less([ni_v, ten_v])
        bb.output(cond_v, np.array(True))
        bb.output(ni_v, i)
        bb.output(nj_v, j)
        bb.output(nk_v, k)

        true_v = gb.const(True)
        oi_v, oj_v, ok_v = gb.Loop(['', true_v, i_v, j_v, k_v],
                                   body=bb.make_graph(),
                                   outputs=['oi', 'oj', 'ok'])
        sum_v = gb.Sum([oi_v, oj_v, ok_v])

        gb.output(sum_v, expected)
        gb.gradient(i_v, np.array(gi, np.float32))
        gb.gradient(j_v, np.array(gj, np.float32))
        gb.gradient(k_v, np.array(gk, np.float32))

        gb.gen_test()

    return fn


# This case needs stacks for retained inputs/outputs in the loop.
def gen_loop_backprop_need_stack_test():
    ii = 1.0
    ji = 1.0
    ki = 1.0
    i = np.array(ii, np.float32)
    j = np.array(ji, np.float32)
    k = np.array(ki, np.float32)
    while i < 100:
        i *= j
        j += 1
        k = np.sqrt(k) * j
    expected = i + j + k

    def fn(test_name):
        gb = onnx_script.GraphBuilder(test_name)
        i = np.array(ii, np.float32)
        j = np.array(ji, np.float32)
        k = np.array(ki, np.float32)
        i_v = gb.param('i', i)
        j_v = gb.param('j', j)
        k_v = gb.param('k', k)

        bb = onnx_script.GraphBuilder(test_name + '_body')
        iter_v = bb.input('iter', np.array(0))
        cond_v = bb.input('cond', np.array(True))
        bi_v = bb.input('bi', i)
        bj_v = bb.input('bj', j)
        bk_v = bb.input('bk', k)
        one_v = bb.const(1.0)
        ni_v = bb.Mul([bi_v, bj_v])
        nj_v = bb.Add([bj_v, one_v])
        nk_v = bb.Mul([bb.Sqrt([bk_v]), nj_v])
        hundred_v = bb.const(100.0)
        cond_v = bb.Less([ni_v, hundred_v])
        bb.output(cond_v, np.array(True))
        bb.output(ni_v, i)
        bb.output(nj_v, j)
        bb.output(nk_v, k)

        true_v = gb.const(True)
        oi_v, oj_v, ok_v = gb.Loop(['', true_v, i_v, j_v, k_v],
                                   body=bb.make_graph(),
                                   outputs=['oi', 'oj', 'ok'])
        sum_v = gb.Sum([oi_v, oj_v, ok_v])

        gb.output(sum_v, expected)
        gb.gradient(i_v, np.array(120.0, np.float32))
        gb.gradient(j_v, np.array(284.1395, np.float32))
        gb.gradient(k_v, np.array(0.7103229, np.float32))

        gb.gen_test()

    return fn


def gen_sequence_test(test_name):
    # TODO(hamaji): Rewrite with onnx_script.
    inputs = [np.array(a) for a in [[1, 2], [3, 4], [5, 6]]]
    nodes = []
    nodes.append(onnx.helper.make_node(
        'ChainerSequenceCreate',
        inputs=[],
        outputs=['seq0']))

    for i, input in enumerate(inputs):
        nodes.append(onnx.helper.make_node(
            'SequenceInsert',
            inputs=['seq%d' % i, 'in%d' % i],
            outputs=['seq%d' % (i + 1)]))

    index_value = 1
    nodes.append(make_constant_node(
        'index', onnx.TensorProto.INT64, [index_value]))
    nodes.append(onnx.helper.make_node(
        'SequenceAt',
        inputs=['seq3', 'index'],
        outputs=['lookup_result']))
    nodes.append(onnx.helper.make_node(
        'ConcatFromSequence',
        inputs=['seq3'],
        outputs=['stack3_result'],
        axis=0,
        new_axis=1))
    nodes.append(onnx.helper.make_node(
        'ConcatFromSequence',
        inputs=['seq2'],
        outputs=['stack2_result'],
        axis=0,
        new_axis=1))
    nodes.append(onnx.helper.make_node(
        'ConcatFromSequence',
        inputs=['seq3'],
        outputs=['concat3_result'],
        axis=0))
    nodes.append(onnx.helper.make_node(
        'ConcatFromSequence',
        inputs=['seq2'],
        outputs=['concat2_result'],
        axis=0))
    nodes.append(onnx.helper.make_node(
        'ChainerSequenceSize',
        inputs=['seq3'],
        outputs=['stack3_size']))

    outputs = [
        ('lookup_result', np.array([3, 4])),
        ('stack3_result', np.stack(inputs)),
        ('stack2_result', np.stack(inputs[0:2])),
        ('concat3_result', np.concatenate(inputs)),
        ('concat2_result', np.concatenate(inputs[0:2])),
        ('stack3_size', np.array(3)),
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


def gen_sequence_pad_test(test_name):
    # TODO(hamaji): Rewrite with GraphBuilder's input/output.
    gb = onnx_script.GraphBuilder(test_name)
    inputs = [np.array(a) for a in [[1, 2, 3], [4], [5, 6]]]
    gb.ChainerSequenceCreate(inputs=[], outputs=['seq0'])

    for i, input in enumerate(inputs):
        gb.SequenceInsert(inputs=['seq%d' % i, 'in%d' % i],
                                outputs=['seq%d' % (i + 1)])

    index_value = 1
    index_v = gb.const([index_value])
    gb.SequenceAt(
        inputs=['seq3', index_v],
        outputs=['lookup_result'])
    gb.ChainerSequencePad(
        value=-42.0,
        length=4,
        inputs=['seq3'],
        outputs=['pad3_result'])
    gb.ChainerSequencePad(
        value=-42.0,
        inputs=['seq2'],
        outputs=['pad2_result'])
    gb.ChainerSequenceLengths(
        inputs=['seq3'],
        outputs=['seq3_lengths_seq'])
    gb.ConcatFromSequence(
        inputs=['seq3_lengths_seq'],
        outputs=['seq3_lengths'],
        axis=0,
        new_axis=1)

    padded = np.array([[1, 2, 3, -42], [4, -42, -42, -42], [5, 6, -42, -42]])
    outputs = [
        ('lookup_result', np.array([4])),
        ('pad3_result', padded),
        ('pad2_result', padded[0:2, 0:3]),
        ('seq3_lengths', np.array([3, 1, 2])),
    ]
    inputs = [('in%d' % i, input) for i, input in enumerate(inputs)]
    inputs_vi = [_extract_value_info(a, n) for n, a in inputs]
    outputs_vi = [_extract_value_info(a, n) for n, a in outputs]
    graph = onnx.helper.make_graph(
        nodes=gb.nodes,
        name=test_name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    gen_test(graph, inputs, outputs, name=test_name)


def gen_sequence_split_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    inputs = np.array([[1, 2, 3, -42], [4, -42, -42, -42], [5, 6, -42, -42]])
    lengths = np.array([3, 1, 2])

    inputs_v = gb.input('input', inputs)
    lengths_v = gb.input('lengths', lengths)

    seq_v = gb.ChainerSequenceSeparate(inputs=[inputs_v], outputs=['seq'])
    lengths_seq_v = gb.ChainerSequenceSeparate(inputs=[lengths_v],
                                           outputs=['lengths_seq'])
    unpadded_v = gb.ChainerSequenceUnpad(inputs=[inputs_v, lengths_seq_v],
                                        outputs=['unpadded'])
    seq_a1_v = gb.ChainerSequenceSeparate(inputs=[inputs_v],
                                      outputs=['seq_a1'],
                                      axis=1)

    for i in range(4):
        index_v = gb.const([i], name='index_%d' % i)
        if i < 3:
            gb.output(gb.SequenceAt(
                inputs=[seq_v, index_v],
                outputs=['split_result_%d' % i]), inputs[i])
            gb.output(gb.SequenceAt(
                inputs=[unpadded_v, index_v],
                outputs=['unpad_result_%d' % i]), inputs[i][:lengths[i]])
        gb.output(gb.SequenceAt(
            inputs=[seq_a1_v, index_v],
            outputs=['split_a1_result_%d' % i]), inputs[:, i])

    gb.gen_test()


def gen_sequence_io_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    input = aranges(3, 2, 4)
    input_seq = [[1, 2, 3, -42], [4, -42, -42, -42], [5, 6, -42, -42]]

    input_v = gb.input('input', input)
    input_seq_v = gb.input('input_seq', Seq(input_seq))

    split_v = gb.ChainerSequenceSeparate([input_v])
    stack_v = gb.ConcatFromSequence([input_seq_v], axis=0, new_axis=1)

    gb.output(gb.Identity([input_v]), input)
    gb.output(gb.Identity([input_seq_v]), Seq(input_seq))
    gb.output(split_v, Seq(list(map(np.squeeze, np.split(input, len(input))))))
    gb.output(stack_v, np.stack(input_seq))

    gb.gen_test()


def gen_sequence_range_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    num_inputs = 0
    for args in [(4,), (-4,), (3, 8), (5, 2),
                 (1, 16, 3), (1, 17, 3), (5, -2, -1), (9, 15, -1)]:
        input_vs = []
        for arg in args:
            input_vs.append(gb.input('input_%d' % num_inputs, arg))
            num_inputs += 1
        output_v = gb.ChainerSequenceRange(input_vs)
        len_v = gb.ChainerSequenceSize([output_v])
        expected = list(range(*args))
        gb.output(len_v, len(expected))
        if expected:
            gb.output(output_v, Seq(expected))
    gb.gen_test()


def gen_sequence_pop_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    inputs = np.array([10, 3, 4, 7, 2, 5])

    inputs_v = gb.input('input', inputs)

    seq_v = gb.ChainerSequenceSeparate(inputs=[inputs_v])
    pop_count = 3
    for i in range(pop_count):
        seq_v, pop_v = gb.ChainerSequencePop(
            inputs=[seq_v],
            outputs=['seq_%d' % i, 'pop_%d' % i]
        )
        gb.output(pop_v, inputs[-1-i])

    # This `seq_v` is used twice, so not-optimized pass will be tested.
    len1_v = gb.ChainerSequenceSize(inputs=[seq_v])
    seq_v, _ = gb.ChainerSequencePop(
        inputs=[seq_v],
        outputs=['seq_final', 'pop_final'],
    )
    len2_v = gb.ChainerSequenceSize(inputs=[seq_v])
    gb.output(gb.Add(inputs=[len1_v, len2_v]),
              (len(inputs) - pop_count) * 2 - 1)

    gb.gen_test()


def gen_sequence_constants_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    inputs = [4, 2, 3]
    seq_v = gb.const_seq(inputs)
    gb.output(seq_v, Seq(inputs))
    gb.gen_test()


def gen_sequence_create_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    inputs = [4, 2, 3]
    inputs_v = [gb.input('input_%d' % i, input)
                for i, input in enumerate(inputs)]
    seq_v = gb.ChainerSequenceCreate(inputs_v)
    stack_v = gb.ConcatFromSequence([seq_v], axis=0, new_axis=1)
    gb.output(seq_v, Seq(inputs))
    gb.output(stack_v, np.array(inputs))
    gb.gen_test()


def gen_sequence_extend_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    input1 = aranges(3, 4)
    input2 = aranges(3, 1) * 2
    seq1 = [np.squeeze(i, 0) for i in np.split(input1, 3)]
    seq2 = [np.squeeze(i, 0) for i in np.split(input2, 3)]

    input1_v = gb.input('input1', input1)
    input2_v = gb.input('input2', input2)
    seq1_v = gb.ChainerSequenceSeparate([input1_v])
    seq2_v = gb.ChainerSequenceSeparate([input2_v])

    gb.output(gb.ChainerSequenceExtend([seq1_v, seq2_v]), Seq(seq1 + seq2))

    gb.gen_test()


def gen_sequence_update_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    inputs = [4, 2, 3]
    seq_v = gb.const_seq(inputs)
    seq_v = gb.ChainerSequenceUpdate([seq_v, gb.const(2), gb.const(42)])
    seq_v = gb.ChainerSequenceUpdate([seq_v, gb.const(-3), gb.const(-49)])
    gb.output(seq_v, Seq([-49, 2, 42]))
    gb.gen_test()


def gen_generic_len_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    input = aranges(4, 2, 3)

    input_v = gb.input('input', input)
    len0_v = gb.ChainerGenericLen([input_v])
    reduced_v = gb.ReduceSum([input_v], axes=[0], keepdims=False)
    len1_v = gb.ChainerGenericLen([reduced_v])
    seq_v = gb.ChainerSequenceSeparate(inputs=[input_v])
    len_seq_v = gb.ChainerGenericLen([seq_v])

    gb.output(len0_v, input.shape[0])
    gb.output(len1_v, input.shape[1])
    gb.output(len_seq_v, input.shape[0])

    gb.gen_test()


def gen_generic_getitem_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    input = aranges(4, 5, 3)
    reduced = np.sum(input, 0)

    input_v = gb.input('input', input)
    reduced_v = gb.ReduceSum([input_v], axes=[0], keepdims=False)
    seq_v = gb.ChainerSequenceSeparate(inputs=[input_v])

    for i in range(-2, 4):
        index_v = gb.const([i])
        gb.output(gb.ChainerGenericGetItem([input_v, index_v]), input[i])
        gb.output(gb.ChainerGenericGetItem([reduced_v, index_v]), reduced[i])
        gb.output(gb.ChainerGenericGetItem([seq_v, index_v]), input[i])

    gb.gen_test()


def gen_generic_getslice_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    input = aranges(4, 5, 3)
    reduced = np.sum(input, 0)

    input_v = gb.input('input', input)
    reduced_v = gb.ReduceSum([input_v], axes=[0], keepdims=False)
    seq_v = gb.ChainerSequenceSeparate(inputs=[input_v])

    def get_slice(input_v, s):
        ins = [input_v]
        if s.start is not None:
            v = gb.const([s.start])
            ins.append(v)
        if s.stop is not None:
            v = gb.const([s.stop])
            ins.append(v)
        if s.step is not None:
            v = gb.const([s.step])
            ins.append(v)
        return gb.ChainerGenericGetSlice(ins)

    def add_test(s):
        expected = input[s]
        gb.output(get_slice(input_v, s), expected)
        gb.output(get_slice(reduced_v, s), reduced[s])
        actual_v = get_slice(seq_v, s)
        if len(expected):
            gb.output(gb.ConcatFromSequence([actual_v], axis=0, new_axis=1),
                      expected)
        else:
            gb.output(gb.ChainerSequenceSize([actual_v]), 0)

    add_test(slice(None))
    for i in range(4):
        add_test(slice(i, None))

    for s, e in [(1, 2), (-2, 3), (0, -2), (999, 9999)]:
        add_test(slice(s, e))

    for s, e, t in [(1, 4, 2), (0, 100, -1), (0, 100, -2)]:
        add_test(slice(s, e, t))

    gb.gen_test()


# TODO(hamaji): Add more tests for both GetItem/SetItem.
def gen_setitem_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    input = np.array([1, 2, 3])

    input_v = gb.input('input', input)
    output_v = gb.ChainerSetItem([input_v, gb.const(1), gb.const(42)],
                                 slice_specs=[1])

    gb.output(output_v, np.array([1, 42, 3]))
    gb.gen_test()


def gen_generic_add_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    input1 = aranges(3, 4)
    input2 = aranges(3, 1) * 2
    seq1 = [np.squeeze(i, 0) for i in np.split(input1, 3)]
    seq2 = [np.squeeze(i, 0) for i in np.split(input2, 3)]

    input1_v = gb.input('input1', input1)
    input2_v = gb.input('input2', input2)
    seq1_v = gb.ChainerSequenceSeparate([input1_v])
    seq2_v = gb.ChainerSequenceSeparate([input2_v])

    gb.output(gb.ChainerGenericAdd([input1_v, input2_v]), input1 + input2)
    gb.output(gb.ChainerGenericAdd([seq1_v, seq2_v]), Seq(seq1 + seq2))
    gb.output(gb.ChainerGenericAdd([input1_v, seq2_v]), input1 + input2)
    gb.output(gb.ChainerGenericAdd([seq1_v, input2_v]), input1 + input2)

    gb.gen_test()


# TODO(hamaji): Test actual output.
def gen_print_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    in1_v = gb.const(21)
    in2_v = gb.const(2)
    result_v = gb.Mul([in1_v, in2_v])
    gb.ChainerPrint([result_v], outputs=[])
    gb.output(gb.Identity([result_v]), 42)
    gb.gen_test()


def gen_hello_world_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    hello = 'Hello, world!\n'
    out_v = gb.ChainerSequenceCreate([])
    for ch in hello:
        ch_v = gb.const(ord(ch), dtype=np.uint8)
        out_v = gb.SequenceInsert([out_v, ch_v])
    gb.output(out_v, Seq(list(np.array(ord(ch), np.uint8) for ch in hello)))
    gb.gen_test()


def gen_type_coersion_test(test_name):
    # Probably, ONNX expects no type coersion happens and this test is
    # not valid ONNX, but we relax the restriction.
    gb = onnx_script.GraphBuilder(test_name)
    iv = 42
    fv = 2.3
    int_v = gb.const(iv)
    float_v = gb.const(fv)

    gb.output(gb.Add([int_v, float_v]), iv + fv)
    gb.output(gb.Add([float_v, int_v]), fv + iv)
    gb.output(gb.Sub([int_v, float_v]), iv - fv)
    gb.output(gb.Sub([float_v, int_v]), fv - iv)
    gb.output(gb.Mul([int_v, float_v]), iv * fv)
    gb.output(gb.Mul([float_v, int_v]), fv * iv)
    gb.output(gb.Div([int_v, float_v]), iv / fv)
    gb.output(gb.Div([float_v, int_v]), fv / iv)

    gb.gen_test()


def gen_incomplete_transpose_test(test_name):
    # ONNX does not allow transposition with incomplete permutations,
    # but this is necessary to realize things like np.swapaxes.
    gb = onnx_script.GraphBuilder(test_name)

    input = aranges(3, 2, 4, 5, 6)
    input_v = gb.input('input', input)
    gb.output(gb.Transpose([input_v], perm=[0, 2, 1]),
              np.transpose(input, axes=[0, 2, 1, 3, 4]))

    gb.gen_test()


def gen_maxpool_cover_all_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)

    input = np.random.random((1, 3, 7, 7))
    input_v = gb.input('input', input)

    # Forget shape.
    squeezed_v = gb.Squeeze([input_v])
    dynamic_v = gb.Unsqueeze([squeezed_v], axes=[0])

    gb.output(gb.MaxPool([input_v], kernel_shape=[3, 3], strides=[2, 2],
                         outputs=['not_cover_all']),
              F.max_pooling_2d(input, ksize=3, stride=2, cover_all=False))
    gb.output(gb.MaxPool([input_v], kernel_shape=[3, 3], strides=[2, 2],
                         ceil_mode=1,
                         outputs=['cover_all']),
              F.max_pooling_2d(input, ksize=3, stride=2, cover_all=True))
    gb.output(gb.MaxPool([dynamic_v], kernel_shape=[3, 3], strides=[2, 2],
                         outputs=['not_cover_all_dynamic']),
              F.max_pooling_2d(input, ksize=3, stride=2, cover_all=False))
    gb.output(gb.MaxPool([dynamic_v], kernel_shape=[3, 3], strides=[2, 2],
                         ceil_mode=1,
                         outputs=['cover_all_dynamic']),
              F.max_pooling_2d(input, ksize=3, stride=2, cover_all=True))

    gb.gen_test()


def gen_batchnorm_training_test(save_mean_var=False):
    def fn(test_name):
        gb = onnx_script.GraphBuilder(test_name)

        batch_size = 2
        chan = 3
        wh = 5
        epsilon = 1e-4
        momentum = 0.95
        chainer.config.train = True

        input = np.random.random((batch_size, chan, wh, wh)).astype(np.float32)
        bn = L.BatchNormalization(chan, decay=momentum, eps=epsilon)
        # Initialize.
        bn(np.random.random((batch_size, chan, wh, wh)).astype(np.float32))

        scale = bn.gamma.array.copy()
        bias = bn.beta.array.copy()
        running_mean = bn.avg_mean.copy()
        running_var = bn.avg_var.copy()

        output = bn(input)

        input_v = gb.input('input', input)
        scale_v = gb.input('scale', scale)
        bias_v = gb.input('bias', bias)
        mean_in_v = gb.input('mean_in', running_mean)
        var_in_v = gb.input('var_in', running_var)

        output_names = ['output', 'mean_out', 'var_out']
        if save_mean_var:
            output_names.extend(['saved_mean', 'saved_var'])

        output_v, mean_out_v, var_out_v, *saved = gb.BatchNormalization(
            inputs=[input_v, scale_v, bias_v, mean_in_v, var_in_v],
            epsilon=epsilon, momentum=momentum,
            outputs=output_names
        )

        gb.output(output_v, output)
        gb.output(mean_out_v, bn.avg_mean)
        gb.output(var_out_v, bn.avg_var)
        if saved:
            mean_out_v, var_out_v = saved
            mean_out = input.mean(axis=(0, 2, 3))
            var_out = input.var(axis=(0, 2, 3))
            np.testing.assert_allclose(output.creator.mean, mean_out)
            gb.output(mean_out_v, mean_out)
            gb.output(var_out_v, var_out)

        gb.gen_test()

    return fn


def gen_spacetodepth_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    small_data = np.array([0.0, 0.1, 0.2, 0.3,
                           1.0, 1.1, 1.2, 1.3,
                           2.0, 2.1, 2.2, 2.3,
                           3.0, 3.1, 3.2, 3.3]).reshape(1, 2, 2, 4)
    input_small = gb.input('input_small', small_data)
    output_small = np.array([0.0, 0.2, 2.0, 2.2,
                             0.1, 0.3, 2.1, 2.3,
                             1.0, 1.2, 3.0, 3.2,
                             1.1, 1.3, 3.1, 3.3]).reshape(1, 8, 1, 2)
    gb.output(gb.SpaceToDepth(
        inputs=['input_small'], blocksize=2, outputs=['output_small']),
        output_small)

    middle_data = np.arange(108, dtype=np.float32).reshape(2, 3, 3, 6)
    input_middle = gb.input('input_middle', middle_data)
    output_middle = np.array([
            0, 3, 18, 21, 36, 39, 1, 4, 19, 22, 37,
            40, 2, 5, 20, 23, 38, 41, 6, 9, 24, 27,
            42, 45, 7, 10, 25, 28, 43, 46, 8, 11, 26,
            29, 44, 47, 12, 15, 30, 33, 48, 51, 13, 16,
            31, 34, 49, 52, 14, 17, 32, 35, 50, 53, 54,
            57, 72, 75, 90, 93, 55, 58, 73, 76, 91, 94,
            56, 59, 74, 77, 92, 95, 60, 63, 78, 81, 96,
            99, 61, 64, 79, 82, 97, 100, 62, 65, 80, 83,
            98, 101, 66, 69, 84, 87, 102, 105, 67, 70, 85,
            88, 103, 106, 68, 71, 86, 89, 104, 107],
        dtype=np.float32).reshape(2, 27, 1, 2)
    gb.output(gb.SpaceToDepth(
        inputs=['input_middle'], blocksize=3, outputs=['output_middle']),
        output_middle)

    gb.gen_test()


def gen_imagescaler_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    test_data = np.ones([2, 3, 4, 5], dtype=np.float32)
    gb.input('input', test_data)
    scale = 2.0
    bias = [1., 2., 3.]
    expected = test_data * scale + np.array(
        bias, dtype=np.float32)[None, :, None, None]

    gb.output(gb.ImageScaler(
        inputs=['input'], scale=scale, bias=bias, outputs=['output']),
        expected)

    gb.gen_test()


def gen_pad_negative_width_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    v = aranges(2, 5, 6, 7)
    i_v = gb.input('input', v)
    gb.output(gb.Pad([i_v], pads=[0, -2, -1, -2, 0, -2, -2, -1]),
              v[:, 2:-2, 1:-2, 2:-1])
    gb.gen_test()


def gen_pad_batch_size_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    v = aranges(2, 5, 6, 7)
    i_v = gb.input('input', v)
    o = np.pad(v, ((0, 6), (0, 0), (0, 0), (0, 0)), 'constant')
    gb.output(gb.ChainerPadBatchSize([i_v], size=8), o)
    gb.gen_test()


def gen_const_int_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    c = np.array(list(range(20)))
    c_v = gb.const(c)
    gb.output(gb.Identity([c_v]), c)
    gb.gen_test()


def gen_const_str_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    i = 42
    i_v = gb.input('input', i)
    c = np.array("hoge", dtype=np.object)
    c_v = gb.const(c)
    gb.ChainerPrint([c_v, i_v], [])
    gb.output(gb.Identity([i_v]), i)
    gb.gen_test()


def gen_const_prop_use_twice_test(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    c = np.array(list(range(20)))
    c_v = gb.const(c)
    gb.output(gb.Add([c_v, c_v]), c * 2)
    gb.gen_test()


def gen_abs_test(dtype):
    def fn(test_name):
        gb = onnx_script.GraphBuilder(test_name)
        i = np.array([42, -24], dtype=dtype)
        i_v = gb.input('input', i)
        gb.output(gb.Abs([i_v]), np.abs(i))
        gb.gen_test()
    return fn


def gen_convtranspose_bn(test_name):
    gb = onnx_script.GraphBuilder(test_name)
    bsize = 2
    ichan = 3
    ochan = 4
    ksize = 3
    isize = 7

    x = aranges(bsize, ochan, isize, isize)
    w = aranges(ochan, ichan, ksize, ksize) * 0.01
    scale = aranges(ichan) * 0.1 + 1
    bias = aranges(ichan) * 0.1 + 2
    mean = aranges(ichan) * 0.1 + 3
    var = aranges(ichan) * 0.1 + 4

    conv = F.deconvolution_2d(x, w, pad=1, outsize=(isize, isize))
    y = F.fixed_batch_normalization(conv, scale, bias, mean, var)

    x_v = gb.input('x', x)
    w_v = gb.param('w', w)
    scale_v = gb.param('scale', scale)
    bias_v = gb.param('bias', bias)
    mean_v = gb.param('mean', mean)
    var_v = gb.param('var', var)

    conv_v = gb.ConvTranspose([x_v, w_v],
                              kernel_shape=[ksize, ksize],
                              pads=[1, 1, 1, 1],
                              output_shape=[isize, isize])
    y_v = gb.BatchNormalization([conv_v, scale_v, bias_v, mean_v, var_v])

    gb.output(y_v, y)
    gb.gen_test()


def gen_unsqueeze_negative_axis(test_name):
    gb = onnx_script.GraphBuilder(test_name)

    x = aranges(1, 2, 3, 5)
    y = np.expand_dims(x, axis=-2)

    x_v = gb.input('x', x)
    y_v = gb.Unsqueeze([x_v], axes=[-2])

    gb.output(y_v, y)
    gb.gen_test()


class TestCase(test_case.TestCase):
    def __init__(self, name, func, **kwargs):
        super(TestCase, self).__init__('out', name, **kwargs)
        self.func = func


def get_tests():
    tests = []
    def test(name, func, diversed=False, **kwargs):
        tests.append(TestCase(name, func, **kwargs))
        if diversed:
            tests.append(TestCase(name + '_diversed', func,
                                  backend='chxvm_test', **kwargs))

    test('extra_test_negative_reshape', gen_negative_reshape_test)

    test('extra_test_inf_nan', gen_inf_nan_test, equal_nan=True)

    test('extra_test_select_item', gen_select_item_test, diversed=True)

    test('extra_test_if_true', gen_if_test(True))
    test('extra_test_if_false', gen_if_test(False))
    test('extra_test_if_with_input_true',
         gen_if_with_input_test(True))
    test('extra_test_if_with_input_false',
         gen_if_with_input_test(False))
    test('extra_test_if_with_external_true',
         gen_if_with_external_test(True))
    test('extra_test_if_with_external_false',
         gen_if_with_external_test(False))

    test('extra_test_loop_basic', gen_loop_test())
    test('extra_test_loop_max_trip_count',
         gen_loop_test(max_trip_count=4))
    test('extra_test_loop_no_max_trip_count',
         gen_loop_test(max_trip_count=None))
    test('extra_test_loop_false_cond',
         gen_loop_test(terminal_condition=False))
    test('extra_test_loop_no_cond',
         gen_loop_test(terminal_condition=None))
    test('extra_test_loop_scan_out',
         gen_loop_test(has_scan_outputs=True))
    test('extra_test_loop_zero_max_trip_count',
         gen_loop_test(max_trip_count=0))
    test('extra_test_loop_zero_trip_count',
         gen_loop_test(cond_trip_count=0))
    # TODO(hamaji): Probably, we do not care loops with zero
    # iterations and scan outputs.
    #
    # TestCase('extra_test_loop_zero_max_trip_count_scan',
    #          gen_loop_test(max_trip_count=0,
    #                        has_scan_outputs=True)),
    # TestCase('extra_test_loop_zero_trip_count_scan',
    #          gen_loop_test(cond_trip_count=0,
    #                        has_scan_outputs=True)),

    test('extra_test_loop_use_enclosing',
                 gen_loop_use_enclosing_test())

    test('extra_backprop_test', gen_backprop_test)

    test('extra_backprop_test_concat', gen_concat_backprop_test)

    test('extra_backprop_test_loop_012',
         gen_loop_backprop_test(0, 1, 2, 1, 5, 1))
    test('extra_backprop_test_loop_000',
         gen_loop_backprop_test(0, 0, 0, 1, 6, 1))
    test('extra_backprop_test_need_stack_loop',
         gen_loop_backprop_need_stack_test())

    test('extra_test_scan_sum', gen_scan_sum_test, fail=True)

    test('extra_test_sequence', gen_sequence_test)
    test('extra_test_sequence_pad', gen_sequence_pad_test)
    test('extra_test_sequence_split', gen_sequence_split_test)
    test('extra_test_sequence_io', gen_sequence_io_test)
    test('extra_test_sequence_range', gen_sequence_range_test)
    test('extra_test_sequence_pop', gen_sequence_pop_test)
    test('extra_test_sequence_constants', gen_sequence_constants_test)
    test('extra_test_sequence_create', gen_sequence_create_test)
    test('extra_test_sequence_extend', gen_sequence_extend_test)
    test('extra_test_sequence_update', gen_sequence_update_test)

    test('extra_test_sentiment_lstm',
         sentiment.gen_rnn_sentiment_test('LSTM'), rtol=0.2)
    test('extra_test_sentiment_bilstm',
         sentiment.gen_rnn_sentiment_test('BiLSTM'),
         rtol=0.5)
    test('extra_test_sentiment_gru',
         sentiment.gen_rnn_sentiment_test('GRU'), rtol=0.4)
    # TODO(hamaji): Investigate why there is a huge error.
    test('extra_test_sentiment_bigru',
         sentiment.gen_rnn_sentiment_test('BiGRU'), rtol=2.5)

    test('extra_test_generic_len', gen_generic_len_test)
    test('extra_test_generic_getitem', gen_generic_getitem_test)
    test('extra_test_generic_getslice', gen_generic_getslice_test)
    test('extra_test_generic_add', gen_generic_add_test)

    test('extra_test_setitem', gen_setitem_test)

    test('extra_test_print', gen_print_test)
    test('extra_test_hello_world', gen_hello_world_test)

    test('extra_test_type_coersion', gen_type_coersion_test,
         skip_shape_inference=True)
    test('extra_test_incomplete_transpose',
         gen_incomplete_transpose_test,
         skip_shape_inference=True)
    test('extra_test_maxpool_cover_all', gen_maxpool_cover_all_test,
         skip_shape_inference=True)

    test('extra_test_batchnorm_training', gen_batchnorm_training_test(False))
    test('extra_test_batchnorm_training_saved',
         gen_batchnorm_training_test(True))

    test('extra_test_spacetodepth', gen_spacetodepth_test)

    test('extra_test_imagescaler', gen_imagescaler_test)

    test('extra_test_pad_negative_width', gen_pad_negative_width_test)

    test('extra_test_pad_batch_size', gen_pad_batch_size_test)

    test('extra_test_const_int', gen_const_int_test)

    test('extra_test_const_str', gen_const_str_test)

    test('extra_test_const_prop_use_twice', gen_const_prop_use_twice_test)

    test('extra_test_abs_int8', gen_abs_test(np.int8))
    test('extra_test_abs_int64', gen_abs_test(np.int64))
    test('extra_test_abs_float16', gen_abs_test(np.float16))

    test('extra_test_convtranspose_bn', gen_convtranspose_bn)

    # TODO(hamaji): ONNX's shape inference for Unsqueeze is probably broken.
    test('extra_test_unsqueeze_negative_axis', gen_unsqueeze_negative_axis,
         skip_shape_inference=True)

    tests += gen_chainercv_op_tests.get_tests()

    return tests


def main():
    for test in get_tests():
        test.func(test.name)


if __name__ == '__main__':
    main()
