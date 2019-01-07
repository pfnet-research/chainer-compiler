# coding: utf-8
# Almost code are from　https://github.com/chainer/onnx-chainer/blob/master/onnx_chainer/testing/test_mxnet.py

import chainer
import numpy as np

import onnx

from chainer import functions as F
from chainer import links as L


def convert_parameter(parameter, name):
    if isinstance(parameter, chainer.Parameter):
        array = parameter.array
    elif isinstance(parameter, chainer.Variable):
        array = parameter.array
    elif isinstance(parameter, np.ndarray):
        array = parameter
    else:
        raise ValueError(
            'The type of parameter is unknown. It should be either Parameter '
            'or Variable or ndarray, but the type was {}.'.format(
                type(parameter)))
    if array.shape == ():
        array = array[None]
    # print('initialize', name, array)
    return onnx.numpy_helper.from_array(array, name)

# 入力xから次元を決める
# モデルにxを流して最初の重みを決める


def collect_inits(lk, pathname):
    res = []
    for na, pa in lk.namedparams():
        if isinstance(pa.data, type(None)):
            continue
        if na.count('/') == 1:
            res.append((pathname + na, pa))

    if isinstance(lk, L.BatchNormalization):
        res.append((pathname + '/avg_mean', lk.avg_mean))
        # TODO(satos) このままだと、nodeのテストは通るがResNetのテストがつらい
        # lk.avg_var = np.ones(lk.avg_var.shape).astype(np.float32) * 4.0
        res.append((pathname + '/avg_var', lk.avg_var))

    elif isinstance(lk, L.NStepLSTM) or isinstance(lk, L.NStepBiLSTM):
        # 先にこちらで集めてしまう
        for i, clk in enumerate(lk.children()):
            for param in clk.params():
                res.append((pathname + '/%d/%s' % (i, param.name), param))
        return res

    for clk in lk.children():
        res += collect_inits(clk, pathname + '/' + clk.name)
    return res


import code


def edit_onnx_protobuf(onnxmod, chainermod):
    initializers = collect_inits(chainermod, '')

    onnx_initializers = []
    for name, param in initializers:
        found = False
        for input in onnxmod.graph.input:
            if input.name == name:
                onnx_initializers.append(convert_parameter(param, name))
                vi = onnx.helper.make_tensor_value_info(
                    'dummy', onnx.TensorProto.FLOAT, param.shape)
                input.type.CopyFrom(vi.type)
                found = True
                break
        assert found, name

    dummygraph = onnx.helper.make_graph(
        [], "hoge", [], [], initializer=onnx_initializers)
    dummygraph.ClearField("name")
    # print(dummygraph)
    onnxmod.graph.MergeFrom(dummygraph)
    return initializers
