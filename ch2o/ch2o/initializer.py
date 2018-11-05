# coding: utf-8
# ほぼ　https://github.com/chainer/onnx-chainer/blob/master/onnx_chainer/testing/test_mxnet.py
# からもらっってきました

import chainer
import numpy as np

from onnx import helper
from onnx import numpy_helper

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
    return numpy_helper.from_array(array, name)

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
    # code.InteractiveConsole({'ch': chainermod}).interact()

    initializers = collect_inits(chainermod, '')
    # print(initializers)
    needparams = list(map(lambda x: x.name, onnxmod.graph.input))
    initializers = list(filter(lambda x: x[0] in needparams, initializers))
    # 真にあるものだけ初期化する

    # code.InteractiveConsole({'ins': initializers, 'onm': onnxmod}).interact()
    tis = list(map(lambda nav: convert_parameter(
        nav[1], nav[0]), initializers))
    dummygraph = helper.make_graph(
        [], "hoge", [], [], initializer=tis)
    dummygraph.ClearField("name")
    # print(dummygraph)
    onnxmod.graph.MergeFrom(dummygraph)
    return initializers
