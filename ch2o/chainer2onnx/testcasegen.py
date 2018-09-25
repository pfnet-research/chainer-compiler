# coding: utf-8
# ほぼ　https://github.com/chainer/onnx-chainer/blob/master/onnx_chainer/testing/test_mxnet.py
# からもらっってきました

import os

import numpy as np

import chainer

from . chainer2onnx import compiler
from . test_args import get_test_args
from . test_args import dprint

from onnx import numpy_helper

from onnx import mapping
from onnx import TensorProto

from . initializer import edit_onnx_protobuf

# variableを消す


def _validate_inout(xs):
    # print(xs)
    if isinstance(xs, chainer.Variable):
        xs = xs.array
    elif isinstance(xs, np.ndarray):
        pass
    elif isinstance(xs, int):
        xs = np.int64(xs)
    elif isinstance(xs, list):
        xs = [_validate_inout(x) for x in xs]
    elif (
            isinstance(xs, np.float32) or
            isinstance(xs, np.float64) or
            isinstance(xs, np.int32) or
            isinstance(xs, np.int64)):
        pass
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
    ys = list(map(lambda y: _validate_inout(y), ys))
    # print('afterys',ys)
    return ys


def dump_test_inputs_outputs(inputs, outputs, test_data_dir):
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    for typ, values in [('input', inputs), ('output', outputs)]:
        for i, (name, value) in enumerate(values):
            if isinstance(value, list):
                assert value
                digits = len(str(len(value)))
                for j, v in enumerate(value):
                    filename = os.path.join(
                        test_data_dir,
                        f'%s_%d_%0{digits}d.pb' % (typ, i, j))
                    tensor = numpy_helper.from_array(v, name)
                    with open(filename, 'wb') as f:
                        f.write(tensor.SerializeToString())
            else:
                filename = os.path.join(test_data_dir,
                                        '%s_%d.pb' % (typ, i))
                tensor = numpy_helper.from_array(value, name)
                with open(filename, 'wb') as f:
                    f.write(tensor.SerializeToString())


def generate_testcase(model, xs, out_key='prob'):
    args = get_test_args()

    # さらの状態からonnxのmodをつくる
    onnxmod, input_tensors, output_tensors = compiler(model)

    chainer.config.train = False
    run_chainer_model(model, xs, out_key)

    dprint("parameter initialized")  # これより前のoverflowは気にしなくて良いはず
    # 1回の実行をもとにinitialize
    edit_onnx_protobuf(onnxmod, model)
    chainer_out = run_chainer_model(model, xs, out_key)

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
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

        xs = list(map(lambda x: _validate_inout(x), xs))

        dump_test_inputs_outputs(
            list(zip(input_names, xs)),
            outputs,
            args.test_data_dir)
