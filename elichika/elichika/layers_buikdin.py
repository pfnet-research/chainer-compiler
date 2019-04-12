import chainer
import chainer.functions as F
import chainer.links as L

import onnx
import onnx.helper as oh
from onnx import numpy_helper
from onnx import TensorProto
from onnx import ModelProto

import elichika.parser.core as core
import elichika.parser.graphs as graphs
import elichika.parser.values as values
import elichika.parser.nodes as nodes
import elichika.parser.functions as functions
import elichika.parser.functions_builtin as functions_builtin
import elichika.parser.values_builtin as values_builtin
import elichika.parser.utils as utils

import numpy as np
import collections

import elichika.onnx_converters as oc

def convert_onnx_chainer_linear(onnx_graph : 'ONNXGraph', node : 'nodes.Node'):
    chainer_inst = node.func.owner.inst # type: chainer.links.Linear
    onnx_name = oc.node2onnx_parameter[node].onnx_name

    x = oc.ONNXValue(onnx_graph, node.inputs[0])
    o = oc.ONNXValue(onnx_graph, node.outputs[0])

    if chainer_inst.W.data is None:
        print("W is unknown. Please infer this model.")

    w = oc.ONNXValue(onnx_graph, chainer_inst.W)

    (x_shape,) = onnx_graph.add_node(
        'Shape',
        [x],
        [None],
        str(node.lineprop))

    (batch_size_1,) = onnx_graph.add_node(
        'Gather',
        [x_shape, oc.ONNXValue(onnx_graph, np.array(0, dtype=np.int64), [onnx_name, '/Zero'])],
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
        [batch_size_2, oc.ONNXValue(onnx_graph, np.array([-1], dtype=np.int64), [onnx_name, '/Minus1'])],
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

def convert_onnx_chainer_convolution2d(onnx_graph : 'ONNXGraph', node : 'nodes.Node'):
    chainer_inst = node.func.owner.inst # type: chainer.links.Convolution2D
    onnx_name = oc.node2onnx_parameter[node].onnx_name

    ksize = oc.size2d(chainer_inst.ksize)
    stride = oc.size2d(chainer_inst.stride)
    ps = oc.size2d(chainer_inst.pad)
    pads = ps + ps

    x = oc.ONNXValue(onnx_graph, node.inputs[0])
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
