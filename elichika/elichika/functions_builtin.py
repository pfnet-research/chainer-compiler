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
import elichika.parser.utils as utils

import numpy as np
import collections

import elichika.onnx_converters as oc


def convert_relu(onnx_graph, node):
    onnx_graph.add_node('Relu',
                        [node.inputs[0]],
                        [node.outputs[0]],
                        name=str(node.lineprop))


def convert_softmax(onnx_graph, node):
    onnx_graph.add_node(
        "Softmax",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        axis=oc.try_get_attribute(node.inputs[1]))


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


def convert_average_pool_2d(onnx_graph, node):
    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    kwargs = {}
    ksize = oc.try_get_attribute(node.inputs[1])
    kwargs['kernel_shape'] = _pair(ksize)

    value = oc.try_get_attribute(node.inputs[2])
    if value is not None:
        kwargs['strides'] = _pair(value)
    else:
        kwargs['strides'] = _pair(ksize)

    value = oc.try_get_attribute(node.inputs[3])
    if value is not None:
        kwargs['pads'] = _pair(value) * 2
    else:
        kwargs['pads'] = _pair(0)

    kwargs['count_include_pad'] = 1

    onnx_graph.add_node(
        "AveragePool",
        [node.inputs[0]],
        [node.outputs[0]],
        name=str(node.lineprop),
        **kwargs,
        )

def convert_unpooling_2d(onnx_graph, node : 'nodes.NodeCall'):
    ksize = oc.try_get_attribute(node.args.keywords['ksize'])
    stride = oc.try_get_attribute(node.args.keywords['stride'])
    pad = oc.try_get_attribute(node.args.keywords['pad'])
    outsize = oc.try_get_attribute(node.args.keywords['outsize'])
    cover_all = oc.try_get_attribute(node.args.keywords['cover_all'])
    
    assert(stride is None) # TODO(hamaji): Not supported yet.
    assert(pad == 0) # TODO(hamaji): Not supported yet.
    assert(outsize is None) # TODO(hamaji): Not supported yet.
    assert(cover_all is False) # TODO(hamaji): Not supported yet.
    
    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    scales = np.array([1, 1] + list(_pair(ksize)), dtype=np.float32)
    scales_ = oc.ONNXValue(onnx_graph, scales, [node, '/Scale'], is_constant = True)
    onnx_graph.add_node(
        "Upsample",
        [node.inputs[0], scales_],
        [node.outputs[0]],
        name=str(node.lineprop))

def convert_reshape(onnx_graph, node):

    onnx_graph.add_node(
        "Reshape",
        [node.inputs[0],oc.ONNXValue(onnx_graph,node.inputs[1]).create_tensor()],
        node.outputs,
        str(node.lineprop))

def convert_split_axis(onnx_graph, node):
    force_tuple = oc.try_get_attribute(node.args.keywords['force_tuple'])
    assert(force_tuple is True) # TODO(hamaji): Not supported yet.

    onnx_graph.add_node(
        "ChainerSequenceSplitAxis",
        [node.inputs[0],oc.ONNXValue(onnx_graph,node.args.keywords['indices_or_sections']).create_tensor()],
        node.outputs,
        str(node.lineprop),
        axis = oc.try_get_attribute(node.args.keywords['axis']))

def convert_roi_max_pooling_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIMaxPool2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value))
    return

def convert_roi_average_pooling_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIAveragePool2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value))
    return

def convert_roi_max_align_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])
    sampling_ratio = oc.ONNXValue(onnx_graph,node.args.keywords['sampling_ratio'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIMaxAlign2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value),
        sampling_ratio=_pair(oc.try_get_attribute(sampling_ratio.value)))
    return

def convert_roi_average_align_2d(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    rois = oc.ONNXValue(onnx_graph,node.args.keywords['rois'])
    roi_indices = oc.ONNXValue(onnx_graph,node.args.keywords['roi_indices'])
    outsize = oc.ONNXValue(onnx_graph,node.args.keywords['outsize'])
    spatial_scale = oc.ONNXValue(onnx_graph,node.args.keywords['spatial_scale'])
    sampling_ratio = oc.ONNXValue(onnx_graph,node.args.keywords['sampling_ratio'])

    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    onnx_graph.add_node(
        "ChainerROIAverageAlign2D",
        [x.create_tensor(), rois.create_tensor(), roi_indices.create_tensor()],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value),
        sampling_ratio=_pair(oc.try_get_attribute(sampling_ratio.value)))
    return
