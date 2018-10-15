# coding: utf-8
# ほぼ　https://github.com/chainer/onnx-chainer/blob/master/onnx_chainer/testing/test_mxnet.py
# からもらっってきました

import collections
import glob
import os
import shutil
import types

import numpy as np

import chainer

from ch2o.chainer2onnx import compiler
from ch2o.test_args import get_test_args
from ch2o.test_args import dprint

from onnx import numpy_helper

from onnx import mapping
from onnx import TensorProto

from ch2o.initializer import edit_onnx_protobuf

# variableを消す


def _validate_inout(xs):
    # print(xs)

    # We use a scalar false as a None.
    # TODO(hamaji): Revisit to check if this decision is OK.
    if xs is None:
        xs = False

    if isinstance(xs, chainer.Variable):
        xs = xs.array
    elif isinstance(xs, np.ndarray):
        pass
    elif isinstance(xs, bool):
        xs = np.array(xs, dtype=np.bool)
    elif isinstance(xs, int):
        xs = np.array(xs, dtype=np.int64)
    elif isinstance(xs, collections.Iterable):
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


def run_chainer_model(model, xs):
    # forward 個分のlistとする
    ys = model(*xs)

    # タプルでなければ 1出力と思う
    if isinstance(ys, tuple):
        ys = list(ys)  # ばらしてみる
    else:
        ys = [ys]

    # print('befys',ys)
    ys = list(map(_validate_inout, ys))
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
                        '%s_%d_%s.pb' % (typ, i, str(j).zfill(digits)))
                    tensor = numpy_helper.from_array(v, name)
                    with open(filename, 'wb') as f:
                        f.write(tensor.SerializeToString())
            else:
                filename = os.path.join(test_data_dir,
                                        '%s_%d.pb' % (typ, i))
                tensor = numpy_helper.from_array(value, name)
                with open(filename, 'wb') as f:
                    f.write(tensor.SerializeToString())


_seen_subnames = set()


def generate_testcase(model, xs, subname=None, has_side_effect=False):
    args = get_test_args()

    def get_model():
        if isinstance(model, type) or isinstance(model, types.FunctionType):
            return model()
        return model

    # さらの状態からonnxのmodをつくる
    onnxmod, input_tensors, output_tensors = compiler(get_model())

    model = get_model()
    chainer.config.train = False
    chainer_out = run_chainer_model(model, xs)

    dprint("parameter initialized")  # これより前のoverflowは気にしなくて良いはず
    # 1回の実行をもとにinitialize
    edit_onnx_protobuf(onnxmod, model)
    if not has_side_effect:
        chainer_out = run_chainer_model(model, xs)

    output_dir = args.output
    if not _seen_subnames:
        # Remove all related directories to renamed tests.
        for d in [output_dir] + glob.glob(output_dir + '_*'):
            if os.path.isdir(d):
                shutil.rmtree(d)
    _seen_subnames.add(subname)
    if subname is not None:
        output_dir = output_dir + '_' + subname
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'model.onnx'), 'wb') as fp:
        fp.write(onnxmod.SerializeToString())

    initializer_names = set()
    for initializer in onnxmod.graph.initializer:
        initializer_names.add(initializer.name)
    input_names = []
    for input_tensor in input_tensors:
        if input_tensor.name not in initializer_names:
            input_names.append(input_tensor.name)

    if len(output_tensors) < len(chainer_out):
        assert len(output_tensors) == 1
        chainer_out = [np.array(chainer_out)]
    assert len(output_tensors) == len(chainer_out)
    outputs = []
    for tensor, value in zip(output_tensors, chainer_out):
        outputs.append((tensor.name, value))

    xs = list(map(lambda x: _validate_inout(x), xs))

    dump_test_inputs_outputs(
        list(zip(input_names, xs)),
        outputs,
        os.path.join(output_dir, 'test_data_set_0'))
