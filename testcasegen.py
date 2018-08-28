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

import code


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

    if isinstance(chainer_out, (list, tuple)):
        chainer_out = [(y.array if isinstance(y,chainer.Variable) else np.array(y)) for y in chainer_out]
    elif isinstance(chainer_out, dict):
        chainer_out = chainer_out[out_key]
        if isinstance(chainer_out, chainer.Variable):
            chainer_out = (chainer_out.array,)
    elif isinstance(chainer_out, chainer.Variable):
        chainer_out = (chainer_out.array,)
    else:
        raise ValueError('Unknown output type: {}'.format(type(chainer_out)))

    return chainer_out


def dump_test_inputs_outputs(inputs, outputs, test_data_dir):
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    for typ, values in [('input', inputs), ('output', outputs)]:
        for i, (name, value) in enumerate(values):
            tensor = numpy_helper.from_array(value, name)
            filename = os.path.join(test_data_dir, '%s_%d.pb' % (typ, i))
            with open(filename, 'wb') as f:
                f.write(tensor.SerializeToString())


from test_initializer import edit_onnx_protobuf

def generate_testcase(model, x, out_key='prob'):
    args = test_args.get_test_args()

    # さらの状態からonnxのmodをつくる
    onnxmod, input_tensors, output_tensors = chainer2onnx.chainer2onnx(
        model, model.forward)
    
    # 生成時にやってる
    # checker.check_model(onnxmod)

    with open(args.raw_output, 'wb') as fp:
        fp.write(onnxmod.SerializeToString())

    chainer.config.train = False
    run_chainer_model(model, x, out_key)

    print("parameter initialized")  # これより前のoverflowは気にしなくて良いはず
    # 1回の実行をもとにinitialize
    edit_onnx_protobuf(onnxmod, model)
    chainer_out = run_chainer_model(model, x, out_key)

    with open(args.output, 'wb') as fp:
        fp.write(onnxmod.SerializeToString())

    if args.test_data_dir:
        initializer_names = set()
        for initializer in onnxmod.graph.initializer:
            initializer_names.add(initializer.name)
        input_names = []
        for input_tensor in input_tensors:
            if input_tensor.name not in initializer_names:
                input_names.append(input_tensor.name)

        # We assume the number of inputs is 1 for now.
        # assert len(input_names) == 1
        # そんなassertはなかった、いいね。

        assert len(output_tensors) == len(chainer_out)
        outputs = []
        for tensor, value in zip(output_tensors, chainer_out):
            outputs.append((tensor.name, value))

        dump_test_inputs_outputs(
            list(zip(input_names, x)),
            outputs,
            args.test_data_dir)

