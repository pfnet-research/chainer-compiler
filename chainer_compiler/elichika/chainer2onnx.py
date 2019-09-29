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
from chainer_compiler.elichika.parser import functions_ndarray
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser import functions_onnx

import numpy as np
import collections
import inspect

from chainer_compiler.elichika import onnx_converters as oc
from chainer_compiler.elichika import links_builtin as lb
from chainer_compiler.elichika import functions_builtin as fb
from chainer_compiler.elichika import functions_chainer_activation as fca


class ONNXModel:
    def __init__(self):
        self.model = None
        self.inputs = []
        self.outputs = []

def validate_args(func, converter):
    if len(inspect.signature(func).parameters) != len(converter.expected_args):
        print("Warning : Mismatch in number of parameters while registering {}".format(func.__name__))
    else:
        for func_arg, converter_arg in zip(inspect.signature(func).parameters, converter.expected_args):
            if func_arg != converter_arg[0]:
                print("Warning : Function argument {} didn't match while registering {}".format(func_arg, func.__name__))


def compile_model(model, inputs) -> 'ONNXModel':

    oc.f_converter.clear()
    oc.chainer_l_converter.clear()

    oc.chainer_l_converter[L.Linear] = lb.convert_onnx_chainer_linear
    oc.chainer_l_converter[L.Convolution2D] = lb.convert_onnx_chainer_convolution2d
    oc.chainer_l_converter[L.ConvolutionND] = lb.convert_onnx_chainer_convolutionnd
    oc.chainer_l_converter[L.BatchNormalization] = lb.convert_onnx_chainer_batch_normalization
    oc.chainer_l_converter[L.NStepLSTM] = lb.convert_onnx_chainer_NStepLSTM
    oc.chainer_l_converter[L.NStepBiLSTM] = lb.convert_onnx_chainer_NStepBiLSTM
    oc.chainer_l_converter[L.EmbedID] = lb.convert_onnx_chainer_EmbedID

    oc.f_converter[F.relu] = fca.ConverterRelu()
    oc.f_converter[F.elu] = fca.ConverterElu()
    oc.f_converter[F.leaky_relu] = fca.ConverterLeakyRelu()
    oc.f_converter[F.log_softmax] = fca.ConverterLogSoftmax()
    oc.f_converter[F.selu] = fca.ConverterSelu()
    oc.f_converter[F.softmax] = fca.ConverterSoftmax()
    oc.f_converter[F.sigmoid] = fca.ConverterSigmoid()

    oc.f_converter[F.pad_sequence] = fb.ConverterPadSequence()
    oc.f_converter[F.softmax_cross_entropy] = fb.ConverterSoftmaxCrossEntropy()
    oc.f_converter[F.average_pooling_2d] = fb.ConverterAveragePool2D()
    oc.f_converter[F.unpooling_2d] = fb.ConverterUnpooling2D()

    oc.f_converter[F.vstack] = fb.ConverterVstack()
    oc.f_converter[F.hstack] = fb.ConverterHstack()
    oc.f_converter[F.stack] = fb.ConverterStack()
    oc.f_converter[F.separate] = fb.ConverterSeparate()
    oc.f_converter[F.squeeze] =  fb.ConverterSqueeze()
    
    oc.f_converter[F.reshape] = fb.ConverterReshape()
    oc.f_converter[F.split_axis] = fb.ConverterSplitAxis()
    oc.f_converter[F.swapaxes] = fb.ConverterSwapaxes()
    oc.f_converter[F.dropout] = fb.ConverterDropout()
    oc.f_converter[F.matmul] = fb.ConverterMatMul()
    oc.f_converter[F.concat] = fb.ConverterConcat()
    oc.f_converter[F.max_pooling_2d] = fb.ConverterMaxPooling2D()
    oc.f_converter[F.resize_images] = fb.ConverterResizeImages()
    oc.f_converter[F.broadcast_to] = fb.ConverterBroadcastTo()
    oc.f_converter[F.expand_dims] = fb.ConverterExpandDims()
    oc.f_converter[F.local_response_normalization] = fb.ConverterResponseNormalization()
    oc.f_converter[F.average] = fb.ConverterAverage()
    oc.f_converter[F.sum] = fb.ConverterSum()
    oc.f_converter[F.maximum] = fb.ConverterChainerMaximum()
    oc.f_converter[F.minimum] = fb.ConverterChainerMinimum()
    oc.f_converter[F.argmax] = fb.ConverterChainerArgMax()
    oc.f_converter[F.argmin] = fb.ConverterChainerArgMin()
    oc.f_converter[F.max] = fb.ConverterMax()
    oc.f_converter[F.min] = fb.ConverterMin()
    oc.f_converter[F.transpose] = fb.ConverterTranspose()

    oc.f_converter[F.sin] = fb.ConverterChainerMathMisc('Sin')
    oc.f_converter[F.sinh] = fb.ConverterChainerMathMisc('Sinh')
    oc.f_converter[F.sign] = fb.ConverterChainerMathMisc('Sign')
    oc.f_converter[F.cos] = fb.ConverterChainerMathMisc('Cos')
    oc.f_converter[F.cosh] = fb.ConverterChainerMathMisc('Cosh')
    oc.f_converter[F.tan] = fb.ConverterChainerMathMisc('Tan')
    oc.f_converter[F.tanh] = fb.ConverterChainerMathMisc('Tanh')
    oc.f_converter[F.arcsin] = fb.ConverterChainerMathMisc('Asin')
    oc.f_converter[F.arccos] = fb.ConverterChainerMathMisc('Acos')
    oc.f_converter[F.arctan] = fb.ConverterChainerMathMisc('Atan')
    oc.f_converter[F.exp] = fb.ConverterChainerMathMisc('Exp')
    oc.f_converter[F.log] = fb.ConverterChainerMathMisc('Log')
    oc.f_converter[F.sqrt] = fb.ConverterChainerMathMisc('Sqrt')
    #oc.f_converter[F.absolute] = fb.ConverterChainerMathMisc('Abs', arg_name='self')

    oc.f_converter[functions_onnx.onnx_abs] = fb.ConverterChainerMathMisc('Abs', arg_name='x')

    oc.f_converter[functions_ndarray.dummy_maximum] = fb.ConverterMaximum()
    oc.f_converter[functions_ndarray.dummy_minimum] = fb.ConverterMinimum()
    oc.f_converter[functions_ndarray.dummy_argmax] = fb.ConverterArgMax()
    oc.f_converter[functions_ndarray.dummy_argmin] = fb.ConverterArgMin()
    oc.f_converter[functions_ndarray.dummy_round] = fb.ConverterRound()
    oc.f_converter[functions_ndarray.dummy_sqrt] = fb.ConverterSqrt()
    oc.f_converter[functions_ndarray.dummy_stack] = fb.ConverterStack()
    oc.f_converter[functions_ndarray.dummy_reshape] = fb.ConverterReshape()
    oc.f_converter[functions_ndarray.dummy_transpose] = fb.ConverterTranspose()

    oc.f_converter[F.clip] = fb.ConverterClip()

    if int(chainer.__version__[0]) >= 6:
        oc.f_converter[F.roi_max_pooling_2d] = fb.ConverterRoiMaxPooling2D()
        oc.f_converter[F.roi_average_pooling_2d] = fb.ConverterRoiAveragePooling2D()
        oc.f_converter[F.roi_max_align_2d] = fb.ConverterRoiMaxAlign2D()

    oc.f_converter[F.roi_average_align_2d] = fb.ConverterRoiAverageAlign2D()

    # validate function args
    for key, value in oc.f_converter.items():
        validate_args(key, value)

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
