# coding: utf-8
# Almost code are from　https://github.com/chainer/onnx-chainer/blob/master/onnx_chainer/testing/test_mxnet.py

import collections
import glob
import os
import shutil
import types

import numpy as np

import chainer

from elichika.chainer2onnx import compile_model, onnx_name
from testtools.test_args import get_test_args
from testtools.test_args import dprint

import onnx
from onnx import numpy_helper
from onnx import TensorProto

from testtools.initializer import edit_onnx_protobuf

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


def validate_chainer_output(ys):
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
        for i, (value_info, value) in enumerate(values):
            name = onnx_name(value_info)
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

                #value_info.type.CopyFrom(onnx.TypeProto())
                #sequence_type = value_info.type.sequence_type
                #tensor_type = sequence_type.elem_type.tensor_type
                #tensor_type.elem_type = tensor.data_type
            else:
                filename = os.path.join(test_data_dir,
                                        '%s_%d.pb' % (typ, i))
                if value is None:
                    if get_test_args().allow_unused_params:
                        continue
                    raise RuntimeError('Unused parameter: %s' % name)
                tensor = numpy_helper.from_array(value, name)
                with open(filename, 'wb') as f:
                    f.write(tensor.SerializeToString())

                vi = onnx.helper.make_tensor_value_info(
                    name, tensor.data_type, tensor.dims)
                #value_info.CopyFrom(vi)


_seen_subnames = set()


def reset_test_generator(args):
    _seen_subnames.clear()
    get_test_args(args)


def generate_testcase(model, xs, subname=None, output_dir=None,
                      backprop=False):
    if output_dir is None:
        args = get_test_args()
        output_dir = args.output

        if backprop:
            output_dir = output_dir + '_backprop'

        if not _seen_subnames:
            # Remove all related directories to renamed tests.
            for d in [output_dir] + glob.glob(output_dir + '_*'):
                if os.path.isdir(d):
                    shutil.rmtree(d)
        assert (backprop, subname) not in _seen_subnames
        _seen_subnames.add((backprop, subname))
        if subname is not None:
            output_dir = output_dir + '_' + subname
    else:
        assert subname is None
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_model():
        if isinstance(model, type) or isinstance(model, types.FunctionType):
            return model()
        return model

    model_ = get_model()
    chainer.config.train = backprop
    model_.cleargrads()
    ys = model_(*xs)
    chainer_out = validate_chainer_output(ys)

    chainer.serializers.save_npz(os.path.join(output_dir, 'chainer_model.npz'), model_)
    model_ = get_model()
    chainer.serializers.load_npz(os.path.join(output_dir, 'chainer_model.npz'), model_)

    onnxmod = compile_model(model_, xs)
    input_tensors = onnxmod.inputs
    output_tensors = onnxmod.outputs

    if backprop:
        ys.grad = np.ones(ys.shape, ys.dtype)
        ys.backward()

    if len(output_tensors) < len(chainer_out):
        assert len(output_tensors) == 1
        chainer_out = [np.array(chainer_out)]
    assert len(output_tensors) == len(chainer_out)

    outputs = list(zip(output_tensors, chainer_out))

    '''
    if backprop:
        for name, param in sorted(model.namedparams()):
            bp_name = onnx.helper.make_tensor_value_info(
                'grad_out@' + name, onnx.TensorProto.FLOAT, ())
            outputs.append((bp_name, param.grad))
    '''

    xs = list(map(lambda x: _validate_inout(x), xs))

    dump_test_inputs_outputs(
        list(zip(input_tensors, xs)),
        outputs,
        os.path.join(output_dir, 'test_data_set_0'))

    with open(os.path.join(output_dir, 'model.onnx'), 'wb') as fp:
        fp.write(onnxmod.model.SerializeToString())
