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


# variableを消す
def unvariable(xs):
    # print(xs)
    if isinstance(xs, chainer.Variable):
        xs = xs.array
    elif isinstance(xs, np.ndarray):
        pass
    elif isinstance(xs, list):
        xs = np.array([unvariable(x) for x in xs])
    else:
        raise ValueError('Unknown type: {}'.format(type(xs)))

    return xs


def run_chainer_model(model, xs, out_key):
    # forward 個分のlistとする
    ys = model(*xs)

    # タプルでなければ 1出力と思う
    if isinstance(ys, tuple):
        ys = list(ys)  # ばらしてみる
    else:
        ys = [ys]

    # print('befys',ys)
    ys = list(map(lambda y: np.array(unvariable(y)), ys))
    # print('afterys',ys)
    return ys


def dump_test_inputs_outputs(inputs, outputs, test_data_dir):
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    for typ, values in [('input', inputs), ('output', outputs)]:
        for i, (name, value) in enumerate(values):
            # とりあえずarrayにする
            # value = unvariable(value)
            if not test_args.get_test_args().quiet:
                print(typ, i, name, value.shape)
                # print(value)
            tensor = numpy_helper.from_array(value, name)
            filename = os.path.join(test_data_dir, '%s_%d.pb' % (typ, i))
            with open(filename, 'wb') as f:
                f.write(tensor.SerializeToString())


from test_initializer import edit_onnx_protobuf


def generate_testcase(model, xs, out_key='prob'):
    args = test_args.get_test_args()

    # さらの状態からonnxのmodをつくる
    onnxmod, input_tensors, output_tensors = chainer2onnx.chainer2onnx(
        model, model.forward)

    with open(args.raw_output, 'wb') as fp:
        fp.write(onnxmod.SerializeToString())

    chainer.config.train = False
    run_chainer_model(model, xs, out_key)

    if not args.quiet:
        print("parameter initialized")  # これより前のoverflowは気にしなくて良いはず
    # 1回の実行をもとにinitialize
    edit_onnx_protobuf(onnxmod, model)
    chainer_out = run_chainer_model(model, xs, out_key)

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

        assert len(output_tensors) == len(chainer_out)
        outputs = []
        for tensor, value in zip(output_tensors, chainer_out):
            outputs.append((tensor.name, value))

        # TODO LSTMのためだがしぶいのでどうにかしたい
        xs = list(map(lambda x: np.array(unvariable(x)), xs))

        dump_test_inputs_outputs(
            list(zip(input_names, xs)),
            outputs,
            args.test_data_dir)
