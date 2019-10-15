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
        assert hasattr(self, 'expected_args'), 'BaseConverter subclass must have `expected_args`'
        parser =  oc.NodeParse()
        for arg_def in self.expected_args:
            parser.add_def(*arg_def)
        parser.parse(onnx_graph, node)
        return parser

    def __call__(self, onnx_graph, node):
        raise NotImplementedError


class ConverterChainerMathMisc(BaseConverter):
    def __init__(self, operator, arg_name = 'x'):
        self.arg_name = arg_name
        self.expected_args = (
            (self.arg_name, oc.ParseType.In),)
        self.operator = operator

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            self.operator,
            [parser.get(self.arg_name)],
            node.outputs,
            name=str(node.lineprop))

def convert_argminmax(onnx_graph, node, parser, tensor, operator, dtype):
    axis = parser.get('axis')

    if axis is None:
        (reshaped,) = onnx_graph.add_node(
        'Reshape',
        [parser.get(tensor), oc.ONNXValue(onnx_graph, np.array([-1], dtype=np.int64), [node, '/Minus1'])],
        [None],
        str(node.lineprop))

        (minmax,) = onnx_graph.add_node(
            operator,
            [reshaped],
            [None],
            str(node.lineprop),
            axis=0)

        (squeesed,) = onnx_graph.add_node(
            'Squeeze',
            [minmax],
            [None],
            str(node.lineprop))

        onnx_graph.add_node(
            "Cast",
            [squeesed],
            node.outputs,
            str(node.lineprop),
            to=get_onnx_dtype(dtype))

    else:
        (minmax,) = onnx_graph.add_node(
            operator,
            [parser.get(tensor)],
            [None],
            str(node.lineprop),
            keepdims=False,
            axis=axis)

        onnx_graph.add_node(
            "Cast",
            [minmax],
            node.outputs,
            str(node.lineprop),
            to=get_onnx_dtype(dtype))

class ConverterChainerArgMax(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            )

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)
        convert_argminmax(onnx_graph, node, parser, 'x', 'ArgMax', np.int32)

class ConverterArgMax(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('a', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            ('out', oc.ParseType.Att, None),
            )

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)
        convert_argminmax(onnx_graph, node, parser, 'a', 'ArgMax', np.int64)

class ConverterChainerArgMin(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            )

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)
        convert_argminmax(onnx_graph, node, parser, 'x', 'ArgMin', np.int32)

class ConverterArgMin(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('a', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            ('out', oc.ParseType.Att, None),
            )

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)
        convert_argminmax(onnx_graph, node, parser, 'a', 'ArgMin', np.int64)

class ConverterChainerMaximum(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x1', oc.ParseType.In),
            ('x2', oc.ParseType.In),
            )

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)
        onnx_graph.add_node(
            "Max",
            [parser.get('x1'), parser.get('x2')],
            node.outputs,
            str(node.lineprop))

class ConverterMaximum(ConverterChainerMaximum):
    def __init__(self):
        super().__init__()
        self.expected_args += (
            ('out', oc.ParseType.Att, None),
            ('where', oc.ParseType.Att, True),
            ('casting', oc.ParseType.Att, 'same_kind'),
            ('order', oc.ParseType.Att, 'K'),
            ('dtype', oc.ParseType.Att, None),
            ('subok', oc.ParseType.Att, True),
            )

class ConverterChainerMinimum(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x1', oc.ParseType.In),
            ('x2', oc.ParseType.In),
            )

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)
        onnx_graph.add_node(
            "Min",
            [parser.get('x1'), parser.get('x2')],
            node.outputs,
            str(node.lineprop))

class ConverterMinimum(ConverterChainerMinimum):
    def __init__(self):
        super().__init__()
        self.expected_args += (
            ('out', oc.ParseType.Att, None),
            ('where', oc.ParseType.Att, True),
            ('casting', oc.ParseType.Att, 'same_kind'),
            ('order', oc.ParseType.Att, 'K'),
            ('dtype', oc.ParseType.Att, None),
            ('subok', oc.ParseType.Att, True),
            )

class ConverterMax(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            ('keepdims', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        axis = parser.get('axis')

        if isinstance(axis, int):
            kwargs['axes'] = [axis]
        elif axis is not None:
            kwargs['axes'] = list(axis)

        onnx_graph.add_node(
            "ReduceMax",
            [parser.get('x')],
            node.outputs,
            str(node.lineprop),
            keepdims=parser.get('keepdims'),
            **kwargs)

class ConverterMin(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            ('keepdims', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        axis = parser.get('axis')

        if isinstance(axis, int):
            kwargs['axes'] = [axis]
        elif axis is not None:
            kwargs['axes'] = list(axis)

        onnx_graph.add_node(
            "ReduceMin",
            [parser.get('x')],
            node.outputs,
            str(node.lineprop),
            keepdims=parser.get('keepdims'),
            **kwargs)

class ConverterClip(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('x_min', oc.ParseType.Att),
            ('x_max', oc.ParseType.Att),)

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "Clip",
            [parser.get('x')],
            node.outputs,
            str(node.lineprop),
            min=parser.get('x_min'),
            max=parser.get('x_max'))

class ConverterSum(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            ('keepdims', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        axis = parser.get('axis')

        if isinstance(axis, int):
            kwargs['axes'] = [axis]
        elif axis is not None:
            kwargs['axes'] = list(axis)

        onnx_graph.add_node(
            "ReduceSum",
            [parser.get('x')],
            node.outputs,
            str(node.lineprop),
            keepdims=parser.get('keepdims'),
            **kwargs)


class ConverterAverage(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            ('weights', oc.ParseType.Att, None),
            ('keepdims', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        axis = parser.get('axis')

        if isinstance(axis, int):
            kwargs['axes'] = [axis]
        elif axis is not None:
            kwargs['axes'] = list(axis)

        onnx_graph.add_node(
            "ReduceMean",
            [parser.get('x')],
            node.outputs,
            str(node.lineprop),
            keepdims=parser.get('keepdims'),
            **kwargs)

class ConverterPadSequence(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('xs', oc.ParseType.In),
            ('length', oc.ParseType.Att),
            ('padding', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        if parser.get('length') is not None:
            kwargs['length'] = parser.get('length')
        if parser.get('padding') is not None:
            kwargs['value'] = float(parser.get('padding'))

        onnx_graph.add_node(
            'ChainerSequencePad',
            [parser.get('xs')],
            node.outputs,
            name=str(node.lineprop),
            **kwargs)


class ConverterDropout(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('ratio', oc.ParseType.Att),
            ('kwargs', oc.ParseType.Ignore))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "Dropout",
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            ratio=parser.get('ratio'))


class ConverterMatMul(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('a', oc.ParseType.In),
            ('b', oc.ParseType.In),
            ('transa', oc.ParseType.Att, False),
            ('transb', oc.ParseType.Att, False))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        assert(parser.get('transa') == False) # TODO(hamaji): Not supported yet.
        assert(parser.get('transb') == False) # TODO(hamaji): Not supported yet.

        onnx_graph.add_node(
            "MatMul",
            [parser.get('a').create_tensor(node.lineprop), parser.get('b').create_tensor(node.lineprop)],
            node.outputs,
            name=str(node.lineprop))


class ConverterConcat(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('xs', oc.ParseType.In),
            ('axis', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ConcatFromSequence",
            [parser.get('xs').create_sequence()],
            node.outputs,
            str(node.lineprop),
            axis=parser.get('axis'))


class ConverterSoftmaxCrossEntropy(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('t', oc.ParseType.In),
            ('normalize', oc.ParseType.Att),
            ('cache_score', oc.ParseType.Att),
            ('class_weight', oc.ParseType.Att),
            ('ignore_label', oc.ParseType.Att),
            ('reduce', oc.ParseType.Att),
            ('enable_double_backprop', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        assert parser.get('normalize')  # TODO(hamaji): Not supported yet.
        assert parser.get('cache_score')  # TODO(hamaji): Not supported yet.
        assert parser.get('class_weight') is None  # TODO(hamaji): Not supported yet.
        assert parser.get('ignore_label') == -1  # TODO(hamaji): Not supported yet.
        assert parser.get('reduce') == 'mean'  # TODO(hamaji): Not supported yet.
        assert not parser.get('enable_double_backprop')  # TODO(hamaji): Not supported yet.

        onnx_graph.add_node(
            "ChainerSoftmaxCrossEntropy",
            [parser.get('x'), parser.get('t')],
            node.outputs,
            str(node.lineprop),
            **kwargs)


class ConverterMaxPooling2D(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('ksize', oc.ParseType.AttPad),
            ('stride', oc.ParseType.AttPad),
            ('pad', oc.ParseType.AttPad),
            ('cover_all', oc.ParseType.Att),
            ('return_indices', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        assert not parser.get('return_indices')  # TODO(hamaji): Not implemented yet.

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

        onnx_graph.add_node(
            "MaxPool",
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            ceil_mode=int(parser.get('cover_all')),
            **kwargs)


class ConverterAveragePool2D(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('ksize', oc.ParseType.AttPad),
            ('stride', oc.ParseType.AttPad),
            ('pad', oc.ParseType.AttPad))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

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
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            **kwargs)

class ConverterUnpooling2D(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('ksize', oc.ParseType.Att),
            ('stride', oc.ParseType.Att),
            ('pad', oc.ParseType.Att),
            ('outsize', oc.ParseType.Att),
            ('cover_all', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        assert(parser.get('stride') is None) # TODO(hamaji): Not supported yet.
        assert(parser.get('pad') == 0) # TODO(hamaji): Not supported yet.
        assert(parser.get('outsize') is None) # TODO(hamaji): Not supported yet.
        assert(parser.get('cover_all') is False) # TODO(hamaji): Not supported yet.

        scales = np.array([1, 1] + list(_pair(parser.get('ksize'))), dtype=np.float32)
        scales_ = oc.ONNXValue(onnx_graph, scales, [node, '/Scale'], is_constant = True)
        onnx_graph.add_node(
            "Upsample",
            [parser.get('x'), scales_],
            node.outputs,
            name=str(node.lineprop),
            **kwargs)


class ConverterResizeImages(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('output_shape', oc.ParseType.Att),
            ('mode', oc.ParseType.Att),
            ('align_corners', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        assert(parser.get('mode') == 'bilinear')  # TODO(hamaji): Not supported yet.
        assert(parser.get('align_corners') == True)  # TODO(hamaji): Not supported yet.

        onnx_graph.add_node(
            "ChainerResizeImages",
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            output_shape=parser.get('output_shape'))


class ConverterVstack(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('xs', oc.ParseType.In),)

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ConcatFromSequence",
            [parser.get('xs').create_sequence()],
            node.outputs,
            name=str(node.lineprop),
            axis=0)


class ConverterHstack(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('xs', oc.ParseType.In),)

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ConcatFromSequence",
            [parser.get('xs').create_sequence()],
            node.outputs,
            name=str(node.lineprop),
            axis=1)


class ConverterStack(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('xs', oc.ParseType.In),
            ('axis', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ChainerSequenceStack",
            [parser.get('xs').create_sequence()],
            node.outputs,
            name=str(node.lineprop),
            axis=parser.get('axis'))


class ConverterSeparate(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ChainerSequenceSeparate",
            [parser.get('x').create_tensor(node.lineprop)],
            node.outputs,
            name=str(node.lineprop),
            axis=parser.get('axis'))


class ConverterSqueeze(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        if parser.get('axis') is not None:
            kwargs['axes'] = _list(parser.get('axis'))

        onnx_graph.add_node(
            "Squeeze",
            [parser.get('x').create_tensor(node.lineprop)],
            node.outputs,
            name=str(node.lineprop),
            **kwargs)


class ConverterReshape(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('shape', oc.ParseType.In))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "Reshape",
            [parser.get('x'), parser.get('shape').create_tensor(node.lineprop)],
            node.outputs,
            name=str(node.lineprop))


class ConverterTranspose(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axes', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "Transpose",
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            perm=parser.get('axes'))


class ConverterSplitAxis(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('indices_or_sections', oc.ParseType.In),
            ('axis', oc.ParseType.Att),
            ('force_tuple', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        assert(parser.get('force_tuple') is True) # TODO(hamaji): Not supported yet.

        onnx_graph.add_node(
            "ChainerSequenceSplitAxis",
            [parser.get('x'), parser.get('indices_or_sections').create_tensor(node.lineprop)],
            node.outputs,
            name=str(node.lineprop),
            axis=parser.get('axis'),
            **kwargs)


class ConverterSwapaxes(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis1', oc.ParseType.Att),
            ('axis2', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        axis1 = parser.get('axis1')
        axis2 = parser.get('axis2')
        pe = list(range(max(axis1, axis2)+1))
        pe[axis1] = axis2
        pe[axis2] = axis1

        onnx_graph.add_node(
            "Transpose",
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            perm=pe)


class ConverterRoiMaxPooling2D(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('rois', oc.ParseType.In),
            ('roi_indices', oc.ParseType.In),
            ('outsize', oc.ParseType.AttPad),
            ('spatial_scale', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ChainerROIMaxPool2D",
            [parser.get('x').create_tensor(node.lineprop), parser.get('rois').create_tensor(node.lineprop), parser.get('roi_indices').create_tensor(node.lineprop)],
            node.outputs,
            str(node.lineprop),
            output_shape=parser.get('outsize'),
            spatial_scale=parser.get('spatial_scale'))


class ConverterRoiAveragePooling2D(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('rois', oc.ParseType.In),
            ('roi_indices', oc.ParseType.In),
            ('outsize', oc.ParseType.AttPad),
            ('spatial_scale', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ChainerROIAveragePool2D",
            [parser.get('x').create_tensor(node.lineprop), parser.get('rois').create_tensor(node.lineprop), parser.get('roi_indices').create_tensor(node.lineprop)],
            node.outputs,
            str(node.lineprop),
            output_shape=parser.get('outsize'),
            spatial_scale=parser.get('spatial_scale'))


class ConverterRoiMaxAlign2D(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('rois', oc.ParseType.In),
            ('roi_indices', oc.ParseType.In),
            ('outsize', oc.ParseType.AttPad),
            ('spatial_scale', oc.ParseType.Att),
            ('sampling_ratio', oc.ParseType.AttPad))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ChainerROIMaxAlign2D",
            [parser.get('x').create_tensor(node.lineprop), parser.get('rois').create_tensor(node.lineprop), parser.get('roi_indices').create_tensor(node.lineprop)],
            node.outputs,
            str(node.lineprop),
            output_shape=parser.get('outsize'),
            spatial_scale=parser.get('spatial_scale'),
            sampling_ratio=parser.get('sampling_ratio'))


class ConverterRoiAverageAlign2D(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('rois', oc.ParseType.In),
            ('roi_indices', oc.ParseType.In),
            ('outsize', oc.ParseType.AttPad),
            ('spatial_scale', oc.ParseType.Att),
            ('sampling_ratio', oc.ParseType.AttPad))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "ChainerROIAverageAlign2D",
            [parser.get('x').create_tensor(node.lineprop), parser.get('rois').create_tensor(node.lineprop), parser.get('roi_indices').create_tensor(node.lineprop)],
            node.outputs,
            str(node.lineprop),
            output_shape=parser.get('outsize'),
            spatial_scale=parser.get('spatial_scale'),
            sampling_ratio=parser.get('sampling_ratio'))


class ConverterBroadcastTo(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('shape', oc.ParseType.In))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "Expand",
            [parser.get('x'), parser.get('shape').create_tensor(node.lineprop)],
            node.outputs,
            name=str(node.lineprop))


class ConverterExpandDims(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('axis', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        onnx_graph.add_node(
            "Unsqueeze",
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            axes=[int(parser.get('axis'))])


class ConverterResponseNormalization(BaseConverter):
    def __init__(self):
        self.expected_args = (
            ('x', oc.ParseType.In),
            ('n', oc.ParseType.Att),
            ('k', oc.ParseType.Att),
            ('alpha', oc.ParseType.Att),
            ('beta', oc.ParseType.Att))

    def __call__(self, onnx_graph, node):
        parser = self.parse_args(onnx_graph, node)

        kwargs = {}
        kwargs['size'] = parser.get('n')
        kwargs['bias'] = float(parser.get('k'))
        kwargs['alpha'] = float(parser.get('alpha') * kwargs['size'])
        kwargs['beta'] = float(parser.get('beta'))

        onnx_graph.add_node(
            "LRN",
            [parser.get('x')],
            node.outputs,
            name=str(node.lineprop),
            **kwargs)
