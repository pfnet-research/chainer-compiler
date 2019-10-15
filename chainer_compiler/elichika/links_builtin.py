import chainer
import chainer.functions as F
import chainer.links as L

import onnx
import onnx.helper as oh
from onnx import numpy_helper
from onnx import TensorProto
from onnx import ModelProto

from chainer_compiler.elichika.parser import core
from chainer_compiler.elichika.parser import graphs
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import functions_builtin
from chainer_compiler.elichika.parser import utils

import numpy as np
import collections

from chainer_compiler.elichika import onnx_converters as oc


def convert_onnx_chainer_linear(onnx_graph: 'ONNXGraph', node: 'nodes.NodeCall'):
    chainer_inst = node.func.owner.inst  # type: chainer.links.Linear
    onnx_name = oc.node2onnx_parameter[node].onnx_name

    x = oc.ONNXValue(onnx_graph, node.args.get_value('x'))
    axes = oc.try_get_attribute(node.args.get_value('n_batch_axes'), node)
    o = oc.ONNXValue(onnx_graph, node.outputs[0])

    if chainer_inst.W.data is None:
        print("W is unknown. Please infer this model.")


    w = oc.ONNXValue(onnx_graph, chainer_inst.W)

    if axes != 1:
        inputs = [x, w]

        if chainer_inst.b is not None:
            b = oc.ONNXValue(onnx_graph, chainer_inst.b)
            inputs.append(b)

        onnx_graph.add_node(
            'ChainerLinear',
            inputs,
            [o],
            str(node.lineprop),
            n_batch_axes=axes)
        return

    (x_shape,) = onnx_graph.add_node(
        'Shape',
        [x],
        [None],
        str(node.lineprop))

    (batch_size_1,) = onnx_graph.add_node(
        'Gather',
        [x_shape, oc.ONNXValue(onnx_graph, np.array(
            0, dtype=np.int64), [onnx_name, '/Zero'])],
        [None],
        str(node.lineprop))

    (batch_size_2,) = onnx_graph.add_node(
        'Unsqueeze',
        [batch_size_1],
        [None],
        str(node.lineprop),
        axes=[0])

    (mat_shape,) = onnx_graph.add_node(
        'Concat',
        [batch_size_2, oc.ONNXValue(onnx_graph, np.array(
            [-1], dtype=np.int64), [onnx_name, '/Minus1'])],
        [None],
        str(node.lineprop),
        axis=0)

    (x_reshape,) = onnx_graph.add_node(
        'Reshape',
        [x, mat_shape],
        [None],
        str(node.lineprop))

    if chainer_inst.b is not None:
        b = oc.ONNXValue(onnx_graph, chainer_inst.b)

        onnx_graph.add_node(
            'Gemm',
            [x_reshape, w, b],
            [o],
            str(node.lineprop),
            transA=0,
            transB=1)
    else:
        temp = oc.ONNXValue(onnx_graph, np.float32, [onnx_name, '/Temp'])

        onnx_graph.add_node(
            'Transpose',
            [w],
            [temp],
            str(node.lineprop),
            perm=[1, 0])

        onnx_graph.add_node(
            'MatMul',
            [x_reshape, temp],
            [o],
            str(node.lineprop))


def convert_onnx_chainer_convolution2d(onnx_graph: 'ONNXGraph', node: 'nodes.NodeCall'):
    chainer_inst = node.func.owner.inst  # type: chainer.links.Convolution2D

    ksize = oc.size2d(chainer_inst.ksize)
    stride = oc.size2d(chainer_inst.stride)
    ps = oc.size2d(chainer_inst.pad)
    pads = ps + ps

    x = oc.ONNXValue(onnx_graph, node.args.get_value('x'))
    o = oc.ONNXValue(onnx_graph, node.outputs[0])
    w = oc.ONNXValue(onnx_graph, chainer_inst.W)
    b = None

    if chainer_inst.b is not None:
        b = oc.ONNXValue(onnx_graph, chainer_inst.b)

    onnx_graph.add_node(
        'Conv',
        [x, w] + ([] if b is None else [b]),
        [o],
        str(node.lineprop),
        kernel_shape=ksize,
        pads=pads,
        strides=stride)


def convert_onnx_chainer_convolutionnd(onnx_graph: 'ONNXGraph', node: 'nodes.NodeCall'):
    chainer_inst = node.func.owner.inst  # type: chainer.links.ConvolutionND

    nd = chainer_inst.W.ndim - 2
    ksize = oc.size_nd(chainer_inst.ksize, nd)
    stride = oc.size_nd(chainer_inst.stride, nd)
    ps = oc.size_nd(chainer_inst.pad, nd)
    pads = ps + ps

    x = oc.ONNXValue(onnx_graph, node.args.get_value('x'))
    o = oc.ONNXValue(onnx_graph, node.outputs[0])
    w = oc.ONNXValue(onnx_graph, chainer_inst.W)
    b = None

    if chainer_inst.b is not None:
        b = oc.ONNXValue(onnx_graph, chainer_inst.b)

    onnx_graph.add_node(
        'Conv',
        [x, w] + ([] if b is None else [b]),
        [o],
        str(node.lineprop),
        kernel_shape=ksize,
        pads=pads,
        strides=stride)


def convert_onnx_chainer_batch_normalization(onnx_graph: 'ONNXGraph', node: 'nodes.NodeCall'):
    chainer_inst = node.func.owner.inst  # type: chainer.links.BatchNormalization

    assert(chainer_inst.axis is None)  # not support yet

    x = oc.ONNXValue(onnx_graph, node.args.get_value('x'))
    o = oc.ONNXValue(onnx_graph, node.outputs[0])

    gamma = oc.ONNXValue(onnx_graph, chainer_inst.gamma)
    beta = oc.ONNXValue(onnx_graph, chainer_inst.beta)
    avg_mean = oc.ONNXValue(onnx_graph, chainer_inst.avg_mean, [node, 'mean'])
    avg_var = oc.ONNXValue(onnx_graph, chainer_inst.avg_var, [node, 'var'])
    eps = chainer_inst.eps
    momentum = chainer_inst.decay

    onnx_graph.add_node(
        'BatchNormalization',
        [x, gamma, beta, avg_mean, avg_var],
        [o],
        str(node.lineprop),
        epsilon=eps,
        momentum=momentum)


def convert_onnx_chainer_NStepLSTM(onnx_graph: 'ONNXGraph', node: 'nodes.NodeCall'):
    chainer_inst = node.func.owner.inst  # type: chainer.links.NStepLSTM

    hd = chainer_inst.children().__next__()

    if not(hd.w0 is None):
        n_in = hd.w0.shape[1]
    else:
        n_in = None

    out_size = chainer_inst.out_size
    n_layers = chainer_inst.n_layers
    dropout = chainer_inst.dropout

    self_ws = []
    self_bs = []
    for i in range(n_layers):
        ws_ = []
        bs_ = []
        for j in range(8):
            ws_.append(oc.ONNXValue(onnx_graph, chainer_inst.ws[i][j]))
            bs_.append(oc.ONNXValue(onnx_graph, chainer_inst.bs[i][j]))
        self_ws.append(ws_)
        self_bs.append(bs_)

    parser = oc.NodeParse()
    parser.add_def('hx', oc.ParseType.Att, None)
    parser.add_def('cx', oc.ParseType.Att, None)
    parser.add_def('xs', oc.ParseType.In)
    parser.parse(onnx_graph, node)

    xs = parser.get('xs').create_sequence()

    # disolve nstep into 1step

    (ilens,) = onnx_graph.add_node(
        'ChainerSequenceLengths',
        [xs],
        [None],
        str(node.lineprop))

    (tilens,) = onnx_graph.add_node(
        'ConcatFromSequence',
        [ilens],
        [None],
        str(node.lineprop),
        axis=0,
        new_axis=True)

    (v,) = onnx_graph.add_node(
        "ChainerSequencePad",
        [xs],
        [None],
        str(node.lineprop))

    (v,) = onnx_graph.add_node(
        "Transpose",
        [v],
        [None],
        str(node.lineprop),
        perm=(1, 0, 2),
    )

    def lstm_param(ps):
        (p,) = onnx_graph.add_node(
            "Concat",
            [v_ for v_ in ps],
            [None],
            str(node.lineprop),
            axis=0
        )

        return onnx_graph.add_node(
            "Unsqueeze",
            [p],
            [None],
            str(node.lineprop),
            axes=[0]
        )[0]

    ws = []
    rs = []
    bs = []
    for w in self_ws:
        ws.append(lstm_param([w[0], w[3], w[1], w[2]]))
        rs.append(lstm_param([w[4], w[7], w[5], w[6]]))

    for b in self_bs:
        bs.append(lstm_param([b[0], b[3], b[1], b[2],
                              b[4], b[7], b[5], b[6]]))

    hs = []
    cs = []
    for i in range(n_layers):
        h = oc.ONNXValue(onnx_graph, np.float32, [node, '/h'])
        c = oc.ONNXValue(onnx_graph, np.float32, [node, '/c'])
        ys = oc.ONNXValue(onnx_graph, np.float32, [node, '/ys'])

        onnx_graph.add_node(
            "LSTM",
            [v, ws[i], rs[i], bs[i], tilens],
            [ys, h, c],
            str(node.lineprop),
            direction='forward',
            hidden_size=out_size,
            # sequence_lens=[ilens.name]
        )

        hs.append(h)
        cs.append(c)
        (yys,) = onnx_graph.add_node(
            "Squeeze",
            [ys],
            [None],
            str(node.lineprop),
            axes=[1]
        )

        v = yys

    onnx_graph.add_node(
        "Concat",
        hs,
        [node.outputs[0]],
        str(node.lineprop),
        axis=0,
    )

    onnx_graph.add_node(
        "Concat",
        cs,
        [node.outputs[1]],
        str(node.lineprop),
        axis=0,
    )

    (tv,) = onnx_graph.add_node(
        "Transpose",
        [v],
        [None],
        str(node.lineprop),
        perm=(1, 0, 2),
    )
    v = tv

    onnx_graph.add_node(
        "ChainerSequenceUnpad",
        [v, ilens],
        [node.outputs[2]],
        str(node.lineprop),
    )


def convert_onnx_chainer_NStepBiLSTM(onnx_graph: 'ONNXGraph', node: 'nodes.NodeCall'):
    chainer_inst = node.func.owner.inst  # type: chainer.links.NStepBiLSTM

    hd = chainer_inst.children().__next__()

    if not(hd.w0 is None):
        n_in = hd.w0.shape[1]
    else:
        n_in = None

    out_size = chainer_inst.out_size
    n_layers = chainer_inst.n_layers
    dropout = chainer_inst.dropout

    self_ws = []
    self_bs = []
    for i in range(n_layers * 2):
        ws_ = []
        bs_ = []
        for j in range(8):
            ws_.append(oc.ONNXValue(onnx_graph, chainer_inst.ws[i][j]))
            bs_.append(oc.ONNXValue(onnx_graph, chainer_inst.bs[i][j]))
        self_ws.append(ws_)
        self_bs.append(bs_)

    parser = oc.NodeParse()
    parser.add_def('hx', oc.ParseType.Att, None)
    parser.add_def('cx', oc.ParseType.Att, None)
    parser.add_def('xs', oc.ParseType.In)
    parser.parse(onnx_graph, node)

    xs = parser.get('xs').create_sequence()

    # disolve nstep into 1step

    (ilens,) = onnx_graph.add_node(
        'ChainerSequenceLengths',
        [xs],
        [None],
        str(node.lineprop))

    (tilens,) = onnx_graph.add_node(
        'ConcatFromSequence',
        [ilens],
        [None],
        str(node.lineprop),
        axis=0,
        new_axis=True)

    v = xs

    def lstm_param(ps):
        (p,) = onnx_graph.add_node(
            "Concat",
            [v_ for v_ in ps],
            [None],
            str(node.lineprop),
            axis=0
        )

        return onnx_graph.add_node(
            "Unsqueeze",
            [p],
            [None],
            str(node.lineprop),
            axes=[0]
        )[0]

    wst = []
    rst = []
    bst = []
    for w in self_ws:
        wst.append(lstm_param([w[0], w[3], w[1], w[2]]))
        rst.append(lstm_param([w[4], w[7], w[5], w[6]]))

    for b in self_bs:
        bst.append(lstm_param([b[0], b[3], b[1], b[2],
                               b[4], b[7], b[5], b[6]]))

    ws = []
    rs = []
    bs = []
    for i in range(n_layers):
        for s, t in [(ws, wst), (rs, rst), (bs, bst)]:
            (temp,) = onnx_graph.add_node(
                "Concat",
                [t[i*2], t[i*2+1]],
                [None],
                str(node.lineprop),
                axis=0
            )

            s.append(temp)

    hs = []
    cs = []

    (v,) = onnx_graph.add_node(
        "ChainerSequencePad",
        [v],
        [None],
        str(node.lineprop))

    (v,) = onnx_graph.add_node(
        "Transpose",
        [v],
        [None],
        str(node.lineprop),
        perm=(1, 0, 2),
    )

    (sequence_length,) = onnx_graph.add_node(
        "ChainerGenericLen", [v], [None], str(node.lineprop),)

    minus1 = oc.ONNXValue(onnx_graph, np.array(-1), [node, '/Minus1'])
    out_size2 = oc.ONNXValue(onnx_graph, np.array(out_size * 2), [node, '/Outputs'])

    (sout_shape,) = onnx_graph.add_node(
        "ChainerSequenceCreate", [sequence_length, minus1, out_size2], [None], str(node.lineprop),)
    sout_shape.onnx_type = oc.ONNXValueType.Sequence

    out_shape = sout_shape.create_tensor(node.lineprop)

    for i in range(n_layers):
        h = oc.ONNXValue(onnx_graph, np.float32, [node, '/h'])
        c = oc.ONNXValue(onnx_graph, np.float32, [node, '/c'])
        ys = oc.ONNXValue(onnx_graph, np.float32, [node, '/ys'])

        onnx_graph.add_node(
            "LSTM",
            [v, ws[i], rs[i], bs[i], tilens],
            [ys, h, c],
            str(node.lineprop),
            direction='bidirectional',
            hidden_size=out_size,
            # sequence_lens=[ilens.name]
        )

        hs.append(h)
        cs.append(c)

        # ys :: [seqlen x 2 x batchsize x hiddensize]
        (v,) = onnx_graph.add_node("Transpose", [ys], [
            None], str(node.lineprop), perm=(0, 2, 1, 3))
        (v,) = onnx_graph.add_node("Reshape", [
            v, out_shape], [None], str(node.lineprop))

    (v,) = onnx_graph.add_node("Transpose", [v.name], [None],
                               str(node.lineprop), perm=(1, 0, 2)
                               )

    (v,) = onnx_graph.add_node("ChainerSequenceUnpad",
                               [v.name, ilens.name], [None], str(node.lineprop))
    v.onnx_type = oc.ONNXValueType.Sequence

    v = v.create_sequence()

    onnx_graph.add_node(
        "Concat",
        hs,
        [node.outputs[0]],
        str(node.lineprop),
        axis=0,
    )

    onnx_graph.add_node(
        "Concat",
        cs,
        [node.outputs[1]],
        str(node.lineprop),
        axis=0,
    )

    onnx_graph.add_node(
        'Identity',
        [v],
        [node.outputs[2]],
        str(node.lineprop))


def convert_onnx_chainer_EmbedID(onnx_graph: 'ONNXGraph', node: 'nodes.NodeCall'):
    chainer_inst = node.func.owner.inst  # type: chainer.links.EmbedID

    n_vocab = chainer_inst.W.shape[0]
    n_out = chainer_inst.W.shape[1]

    w = oc.ONNXValue(onnx_graph, chainer_inst.W)

    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.parse(onnx_graph, node)

    x = parser.get('x').create_tensor(node.lineprop)

    onnx_graph.add_node(
        'Gather',
        [w, x],
        [node.outputs[0]],
        str(node.lineprop))
