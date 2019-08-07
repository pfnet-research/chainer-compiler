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


def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)


def _list(v) -> 'List[int]':
    if isinstance(v, collections.Iterable):
        return list(x for x in v)
    return [v]


def get_onnx_dtype(dtype):
    a = np.zeros((), dtype=dtype)
    dt = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[a.dtype]
    return dt


class BaseConverter(object):
    def __init__(self):
        self.expected_args = ()

    def parse_args(self, onnx_graph, node):
        assert hasattr(
            self, 'expected_args'), 'BaseConverter subclass must have `expected_args`'
        parser = oc.NodeParse()
        for arg_def in self.expected_args:
            parser.add_def(*arg_def)
        parser.parse(onnx_graph, node)
        return parser

    def __call__(self, onnx_graph, node):
        raise NotImplementedError


class ConverterRelu(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),)

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            'Relu',
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop))


class ConverterElu(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('alpha', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            'Elu',
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            alpha=parser.get('alpha'))


class ConverterLeakyRelu(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('slope', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            'LeakyRelu',
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            alpha=parser.get('slope'))


class ConverterSelu(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('alpha', oc.ParseType.Att),
            ('scale', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            'Selu',
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            alpha=parser.get('alpha'),
            gamma=parser.get('scale'),
        )


class ConverterSigmoid(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),)

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            'Sigmoid',
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop))


class ConverterSoftmax(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            'Softmax',
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            axis=parser.get('axis'),
            chainer_is_onnx_semantics=False)
