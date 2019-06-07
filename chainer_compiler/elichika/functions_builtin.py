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


def convert_relu(onnx_graph, node):
    onnx_graph.add_node('Relu',
                        [node.inputs[0]],
                        [node.outputs[0]],
                        name=str(node.lineprop))
    return

def convert_tanh(onnx_graph, node):
    onnx_graph.add_node('Tanh',
                        [node.inputs[0]],
                        [node.outputs[0]],
                        name=str(node.lineprop))
    return

def convert_sigmoid(onnx_graph, node):
    onnx_graph.add_node("Sigmoid",
                        [node.inputs[0]],
                        [node.outputs[0]],
                        name=str(node.lineprop))
    return


def convert_sum(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.add_def('keepdims', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    kwargs = {}

    axis = parser.get('axis')

    if isinstance(axis, int):
        kwargs['axes'] = [axis]
    elif axis is not None:
        kwargs['axes'] = list(axis)

    onnx_graph.add_node(
        "ReduceSum",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        keepdims=parser.get('keepdims'),
        **kwargs)

def convert_average(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.add_def('weights', oc.ParseType.Att, None)
    parser.add_def('keepdims', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    kwargs = {}

    axis = parser.get('axis')

    if isinstance(axis, int):
        kwargs['axes'] = [axis]
    elif axis is not None:
        kwargs['axes'] = list(axis)

    onnx_graph.add_node(
        "ReduceMean",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        keepdims=parser.get('keepdims'),
        **kwargs)


def convert_softmax(onnx_graph, node):
    onnx_graph.add_node(
        "Softmax",
        [node.inputs[0]],
        [node.outputs[0]],
        str(node.lineprop),
        axis=oc.try_get_attribute(node.inputs[1]),
        chainer_is_onnx_semantics=False)


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


def convert_dropout(onnx_graph, node):
    x = oc.ONNXValue(onnx_graph,node.args.keywords['x'])
    ratio = oc.try_get_attribute(node.attribute_args.keywords['ratio'])

    onnx_graph.add_node(
        "Dropout",
        [x],
        node.outputs,
        str(node.lineprop),
        ratio=ratio,
        )


def convert_matmul(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('a', oc.ParseType.In)
    parser.add_def('b', oc.ParseType.In)
    parser.add_def('transa', oc.ParseType.Att, False)
    parser.add_def('transb', oc.ParseType.Att, False)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "MatMul",
        [parser.get('a').create_tensor(node.lineprop), parser.get('b').create_tensor(node.lineprop)],
        node.outputs,
        str(node.lineprop),
        )

def convert_concat(onnx_graph, node):
    xs = oc.ONNXValue(onnx_graph,node.args.keywords['xs'])
    axis = oc.try_get_attribute(node.attribute_args.keywords['axis'])

    onnx_graph.add_node(
        "ChainerSequenceConcat",
        [xs.create_sequence()],
        node.outputs,
        str(node.lineprop),
        axis=axis,
        )

def convert_softmax_cross_entropy(onnx_graph, node):
    normalize = oc.try_get_attribute(node.attribute_args.keywords['normalize'])
    cache_score = oc.try_get_attribute(node.attribute_args.keywords['cache_score'])
    class_weight = oc.try_get_attribute(node.attribute_args.keywords['class_weight'])
    ignore_label = oc.try_get_attribute(node.attribute_args.keywords['ignore_label'])
    reduce = oc.try_get_attribute(node.attribute_args.keywords['reduce'])
    enable_double_backprop = oc.try_get_attribute(node.attribute_args.keywords['enable_double_backprop'])

    assert normalize  # TODO(hamaji): Not supported yet.
    assert cache_score  # TODO(hamaji): Not supported yet.
    assert class_weight is None  # TODO(hamaji): Not supported yet.
    assert ignore_label == -1  # TODO(hamaji): Not supported yet.
    assert reduce == 'mean'  # TODO(hamaji): Not supported yet.
    assert not enable_double_backprop  # TODO(hamaji): Not supported yet.

    onnx_graph.add_node(
        "ChainerSoftmaxCrossEntropy",
        node.inputs[0:2],
        node.outputs,
        str(node.lineprop))

def convert_max_pooling_2d(onnx_graph, node):
    def _pair(x):
        if isinstance(x, collections.Iterable):
            return x
        return (x, x)

    ksize = oc.try_get_attribute(node.attribute_args.keywords['ksize'])
    stride = oc.try_get_attribute(node.attribute_args.keywords['stride'])
    pad = oc.try_get_attribute(node.attribute_args.keywords['pad'])
    cover_all = oc.try_get_attribute(node.attribute_args.keywords['cover_all'])
    return_indices = oc.try_get_attribute(node.attribute_args.keywords['return_indices'])

    assert not return_indices  # TODO(hamaji): Not implemented yet.

    kwargs = {}
    kwargs['kernel_shape'] = _pair(ksize)

    if stride is not None:
        kwargs['strides'] = _pair(stride)
    else:
        kwargs['strides'] = _pair(ksize)

    if pad is not None:
        kwargs['pads'] = _pair(pad) * 2
    else:
        kwargs['pads'] = _pair(0)

    onnx_graph.add_node(
        "MaxPool",
        [node.inputs[0]],
        [node.outputs[0]],
        name=str(node.lineprop),
        chainer_cover_all=cover_all,
        **kwargs,
        )

def convert_average_pool_2d(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.add_def('ksize', oc.ParseType.AttPad)
    parser.add_def('stride', oc.ParseType.AttPad)
    parser.add_def('pad', oc.ParseType.AttPad)
    parser.parse(onnx_graph, node)


    kwargs = {}

    kwargs['kernel_shape'] = parser.get('ksize')
    if parser.get('stride') is not None:
        kwargs['strides'] = parser.get('stride')
    else:
        kwargs['strides'] = parser.get('ksize')
    if parser.get('pad') is not None:
        kwargs['pads'] = parser.get('pad') *2
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
    ksize = oc.try_get_attribute(node.attribute_args.keywords['ksize'])
    stride = oc.try_get_attribute(node.attribute_args.keywords['stride'])
    pad = oc.try_get_attribute(node.attribute_args.keywords['pad'])
    outsize = oc.try_get_attribute(node.attribute_args.keywords['outsize'])
    cover_all = oc.try_get_attribute(node.attribute_args.keywords['cover_all'])

    assert(stride is None) # TODO(hamaji): Not supported yet.
    assert(pad == 0) # TODO(hamaji): Not supported yet.
    assert(outsize is None) # TODO(hamaji): Not supported yet.
    assert(cover_all is False) # TODO(hamaji): Not supported yet.

    scales = np.array([1, 1] + list(_pair(ksize)), dtype=np.float32)
    scales_ = oc.ONNXValue(onnx_graph, scales, [node, '/Scale'], is_constant = True)
    onnx_graph.add_node(
        "Upsample",
        [node.inputs[0], scales_],
        [node.outputs[0]],
        name=str(node.lineprop))

def convert_resize_images(onnx_graph, node):
    output_shape = oc.try_get_attribute(node.attribute_args.keywords['output_shape'])

    onnx_graph.add_node(
        "ChainerResizeImages",
        [node.inputs[0]],
        [node.outputs[0]],
        name=str(node.lineprop),
        output_shape=_pair(output_shape))

def convert_vstack(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('xs', oc.ParseType.In)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceConcat",
        [parser.get('xs').create_sequence()],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=0)

def convert_hstack(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('xs', oc.ParseType.In)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceConcat",
        [parser.get('xs').create_sequence()],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=1)

def convert_stack(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('xs', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceStack",
        [parser.get('xs').create_sequence()],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=parser.get('axis'))

def convert_separate(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    onnx_graph.add_node(
        "ChainerSequenceSeparate",
        [parser.get('x').create_tensor(node.lineprop)],
        [node.outputs[0]],
        name=str(node.lineprop),
        axis=parser.get('axis'))

def convert_squeeze(onnx_graph, node):
    parser = oc.NodeParse()
    parser.add_def('x', oc.ParseType.In)
    parser.add_def('axis', oc.ParseType.Att)
    parser.parse(onnx_graph, node)

    kwargs = {}
    if parser.get('axis') is not None:
        kwargs['axes'] = _list(parser.get('axis'))

    onnx_graph.add_node(
        "Squeeze",
        [parser.get('x').create_tensor(node.lineprop)],
        [node.outputs[0]],
        name=str(node.lineprop),
        **kwargs)

def convert_reshape(onnx_graph, node):
    onnx_graph.add_node(
        "Reshape",
        [node.inputs[0],oc.ONNXValue(onnx_graph,node.inputs[1]).create_tensor(node.lineprop)],
        node.outputs,
        str(node.lineprop))

def convert_split_axis(onnx_graph, node):
    force_tuple = oc.try_get_attribute(node.attribute_args.keywords['force_tuple'])
    assert(force_tuple is True) # TODO(hamaji): Not supported yet.

    onnx_graph.add_node(
        "ChainerSequenceSplitAxis",
        [node.inputs[0],oc.ONNXValue(onnx_graph,node.args.keywords['indices_or_sections']).create_tensor(node.lineprop)],
        node.outputs,
        str(node.lineprop),
        axis = oc.try_get_attribute(node.attribute_args.keywords['axis']))

def convert_swapaxes(onnx_graph, node):
    axis1 = oc.try_get_attribute(node.attribute_args.keywords['axis1'])
    axis2 = oc.try_get_attribute(node.attribute_args.keywords['axis2'])
    pe = list(range(max(axis1, axis2)+1))
    pe[axis1] = axis2
    pe[axis2] = axis1

    onnx_graph.add_node(
        "Transpose",
        [node.inputs[0]],
        node.outputs,
        str(node.lineprop),
        perm = pe)

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
        [x.create_tensor(node.lineprop), rois.create_tensor(node.lineprop), roi_indices.create_tensor(node.lineprop)],
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
        [x.create_tensor(node.lineprop), rois.create_tensor(node.lineprop), roi_indices.create_tensor(node.lineprop)],
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
        [x.create_tensor(node.lineprop), rois.create_tensor(node.lineprop), roi_indices.create_tensor(node.lineprop)],
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
        [x.create_tensor(node.lineprop), rois.create_tensor(node.lineprop), roi_indices.create_tensor(node.lineprop)],
        node.outputs,
        str(node.lineprop),
        output_shape=_pair(oc.try_get_attribute(outsize.value)),
        spatial_scale=oc.try_get_attribute(spatial_scale.value),
        sampling_ratio=_pair(oc.try_get_attribute(sampling_ratio.value)))
    return

def convert_broadcast_to(onnx_graph, node):
    node_ = node

    shape = oc.ONNXValue(onnx_graph, node_.args.keywords['shape'])
    onnx_graph.add_node(
        "Expand",
        [node_.inputs[0], shape.create_tensor(node.lineprop)],
        node_.outputs,
        str(node.lineprop))
    return


def convert_expand_dims(onnx_graph, node):
    node_ = node
    axis = oc.try_get_attribute(node_.args.keywords['axis'])
    onnx_graph.add_node(
        'Unsqueeze',
        [node_.inputs[0]],
        node_.outputs,
        str(node.lineprop),
        axes=[int(axis)])
    return

def convert_local_response_normalization(onnx_graph, node):
    kwargs = {}
    kwargs['size'] = oc.try_get_attribute(node.attribute_args.keywords['n'])
    kwargs['bias'] = float(oc.try_get_attribute(node.attribute_args.keywords['k']))
    kwargs['alpha'] = float(oc.try_get_attribute(node.attribute_args.keywords['alpha']) * kwargs['size'])
    kwargs['beta'] = float(oc.try_get_attribute(node.attribute_args.keywords['beta']))

    onnx_graph.add_node(
        "LRN",
        [node.inputs[0]],
        node.outputs,
        str(node.lineprop),
        **kwargs,
    )
