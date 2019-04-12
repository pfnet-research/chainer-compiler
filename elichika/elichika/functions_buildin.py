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

def convert_relu(onnx_graph, node):
    onnx_graph.add_node('Relu', 
        [node.inputs[0]],
        [node.outputs[0]],
        name = str(node.lineprop))

def convert_softmax(onnx_graph, node):
    onnx_graph.add_node(
        "Softmax",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        axis = oc.try_get_attribute(node.inputs[1]))

def convert_pad_sequence(onnx_graph, node):
    kwargs = {}

    if node.inputs[1] is not None:
        value = oc.try_get_attribute(node.inputs[1])
        if value is not None:
            kwargs['length'] = value
        if node.inputs[2] is not None:
            value = oc.try_get_attribute(node.inputs[2])
            if value != 0:
                kwargs['value'] = float(value)

    onnx_graph.add_node(
        "ChainerSequencePad",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        **kwargs)

def convert_softmax_cross_entropy(onnx_graph, node):

    onnx_graph.add_node(
        "ChainerSoftmaxCrossEntropy",
        node.inputs,
        node.outputs,
        str(node.lineprop))
