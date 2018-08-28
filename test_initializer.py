# coding: utf-8
# ほぼ　https://github.com/chainer/onnx-chainer/blob/master/onnx_chainer/testing/test_mxnet.py
# からもらっってきました

import collections
import os
import warnings

import numpy as np

import chainer
import chainer2onnx
import test_args

from onnx import checker
from onnx import helper
from onnx import numpy_helper

from chainer import links as L
from chainer import functions as F

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


import code

def collect_inits(lk,pathname):
    res = []
    for na,pa in lk.namedparams():
        if isinstance(pa.data, type(None)):
            continue
        if na.count('/')==1:
            res.append((pathname + '_' + na[1:],pa))
    
    if isinstance(lk,L.BatchNormalization):
        res.append((pathname + '_avg_mean',lk.avg_mean))
        v = lk.avg_var
        v.array = np.ones(v.shape).astype(np.float32) * 4.0
        res.append((pathname + '_avg_var',v))
    elif isinstance(lk,L.NStepLSTM):
        # 先にこちらで集めてしまう
        for i,clk in enumerate(lk.children()):
            #for t in range(4,8):
            #    exec("clk.w%d.array.fill(2)" % t)
            
            #どーにも, linkの w2 が式中の w3 に見えるんだが...???
            #clk.w0.array.fill(1)

                # これだと壊れるので .array に入れましょう 
                # exec("clk.w%d = np.zeros(clk.w%d.shape).astype(np.float32)" % (t,t))
            
            
            #for p in clk.params():
            #    p.array.fill(0)
            #code.InteractiveConsole({'lk': clk}).interact()
            
            ws0 = F.concat([clk.w0.T,clk.w3.T,clk.w1.T,clk.w2.T]).T
            #ws0 = F.stack([clk.w0,clk.w3,clk.w1,clk.w2])
            ws1 = F.concat([clk.w4.T,clk.w7.T,clk.w5.T,clk.w6.T]).T
            
            #print('shape',ws0.shape)
            #>>> ','.join(map(lambda s: 'clk.b%d' % s,range(8)))
            #bss = F.hstack([clk.b0,clk.b1,clk.b2,clk.b3,clk.b4,clk.b5,clk.b6,clk.b7])
            bss = F.hstack([clk.b0,clk.b3,clk.b1,clk.b2,clk.b4,clk.b7,clk.b5,clk.b6])
            #print('cs',ws0,ws1,bss) 
            ws0 = np.array([ws0.data])
            ws1 = np.array([ws1.data])
            bss = np.array([bss.data])
            # Uni-directional なので全部1次元になる
            res.append((pathname + '_%d_ws0' % i,ws0))
            res.append((pathname + '_%d_ws1' % i,ws1))
            res.append((pathname + '_%d_bss' % i,bss))
            
        return res

    for clk in lk.children():
        res += collect_inits(clk,pathname + '_' + clk.name)
    return res

def edit_onnx_protobuf(onnxmod, chainermod):
    # code.InteractiveConsole({'ch': chainermod}).interact()
    
    initializers = collect_inits(chainermod,'')
    #print(initializers) 
    
    tis = list(map(lambda nav: convert_parameter(nav[1],nav[0]),initializers))
    dummygraph = helper.make_graph(
        [], "hoge", [], [], initializer=tis)
    dummygraph.ClearField("name")
    # print(dummygraph)
    onnxmod.graph.MergeFrom(dummygraph)
    return initializers


def run_chainer_model(model, x, out_key):
    # Forward computation
    if isinstance(x, (list, tuple)):
        #for i in x:
        #    assert isinstance(i, (np.ndarray, chainer.Variable))

        #LSTMとかの場合、これはfailするので無視する
        chainer_out = model(*x)
    elif isinstance(x, np.ndarray):
        chainer_out = model(chainer.Variable(x))
    elif isinstance(x, chainer.Variable):
        chainer_out = model(x)
    else:
        raise ValueError(
            'The \'x\' argument should be a list or tuple of numpy.ndarray or '
            'chainer.Variable, or simply numpy.ndarray or chainer.Variable '
            'itself. But a {} object was given.'.format(type(x)))

    # print(chainer_out)
    # code.InteractiveConsole({'co': chainer_out}).interact()

    return chainer_out

