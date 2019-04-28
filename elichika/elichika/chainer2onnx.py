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
import elichika.links_builtin as lb
import elichika.functions_builtin as fb


class ONNXModel:
    def __init__(self):
        self.model = None
        self.inputs = []
        self.outputs = []


def compile_model(model, inputs) -> 'ONNXModel':

    oc.chainer_f_converter.clear()
    oc.chainer_l_converter.clear()

    oc.chainer_l_converter[L.Linear] = lb.convert_onnx_chainer_linear
    oc.chainer_l_converter[L.Convolution2D] = lb.convert_onnx_chainer_convolution2d
    oc.chainer_l_converter[L.BatchNormalization] = lb.convert_onnx_chainer_batch_normalization

    oc.chainer_f_converter[F.relu] = fb.convert_relu
    oc.chainer_f_converter[F.softmax] = fb.convert_softmax
    oc.chainer_f_converter[F.pad_sequence] = fb.convert_pad_sequence
    oc.chainer_f_converter[F.softmax_cross_entropy] = fb.convert_softmax_cross_entropy
    oc.chainer_f_converter[F.average_pooling_2d] = fb.convert_average_pool_2d
    oc.chainer_f_converter[F.unpooling_2d] = fb.convert_unpooling_2d
    oc.chainer_f_converter[F.reshape] = fb.convert_reshape
    oc.chainer_f_converter[F.split_axis] = fb.convert_split_axis
    oc.chainer_f_converter[F.swapaxes] = fb.convert_swapaxes
    oc.chainer_f_converter[F.dropout] = fb.convert_dropout
    oc.chainer_f_converter[F.matmul] = fb.convert_matmul
    oc.chainer_f_converter[F.concat] = fb.convert_concat
    oc.chainer_f_converter[F.max_pooling_2d] = fb.convert_max_pooling_2d
    oc.chainer_f_converter[F.resize_images] = fb.convert_resize_images
    oc.chainer_f_converter[F.tanh] = fb.convert_tanh
    oc.chainer_f_converter[F.sigmoid] = fb.convert_sigmoid
    oc.chainer_f_converter[F.broadcast_to] = fb.convert_broadcast_to
    oc.chainer_f_converter[F.expand_dims] = fb.convert_expand_dims

    if int(chainer.__version__[0]) >= 6:
        oc.chainer_f_converter[F.roi_max_pooling_2d] = fb.convert_roi_max_pooling_2d
        oc.chainer_f_converter[F.roi_average_pooling_2d] = fb.convert_roi_average_pooling_2d
        oc.chainer_f_converter[F.roi_max_align_2d] = fb.convert_roi_max_align_2d

    oc.chainer_f_converter[F.roi_average_align_2d] = fb.convert_roi_average_align_2d

    # assign names
    oc.assigned_names.clear()
    oc.node2onnx_parameter.clear()
    oc.value2onnx_parameter.clear()

    inputs_, outputs_, graph_ = core.convert_model(model, inputs)

    if graph_ is None:
        return None

    oc.preprocess(graph_, True)

    generator = oc.ONNXGenerator()
    model = generator.generate_model(
        graph_.input_values, graph_.output_values, graph_, model)

    # check inputs

    onnx_model = ONNXModel()
    onnx_model.model = model
    onnx_model.inputs = graph_.input_values
    onnx_model.outputs = graph_.output_values
    return onnx_model


def save_model(path: 'str', model: 'ModelProto'):
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def save_model_as_text(path: 'str', model: 'ModelProto'):
    with open(path, "w") as f:
        print(model, file=f)
