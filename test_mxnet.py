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

try:
    import mxnet
    MXNET_AVAILABLE = True
except ImportError:
    warnings.warn(
        'MXNet is not installed. Please install mxnet to use '
        'testing utility for compatiblity checking.',
        ImportWarning)
    MXNET_AVAILABLE = False


from onnx import checker
from onnx import helper
from onnx import numpy_helper


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


# import code


def edit_onnx_protobuf(onnxmod, x, chainermod):
    # code.InteractiveConsole({'ch': chainermod}).interact()

    initializers = []

    avg_pass = []
    for na, pa in chainermod.namedparams():
        if isinstance(pa.data, type(None)):
            continue
        # print(na)
        initializers.append(convert_parameter(pa, na.replace('/', '_')))
        # これだとbatchnormの avg_mean と avg_var が入らない
        if na[-5:] == '/beta':
            avg_pass.append(na[:-5])

    # めちゃアドホックなのでどうにかしたい
    def rec(v, p):
        if len(p) == 0:
            return v
        else:
            if p[0].isdigit():
                return rec(v[int(p[0])], p[1:])
            else:
                return rec(getattr(v, p[0]), p[1:])

    for p in avg_pass:
        # print(p)
        # code.InteractiveConsole({'ch': chainermod}).interact()
        na = p + '/avg_mean'
        v = rec(chainermod, na.split('/')[1:])
        # print(na,v)
        initializers.append(convert_parameter(v, na.replace('/', '_')))
        na = p + '/avg_var'
        v = rec(chainermod, na.split('/')[1:])

        # これvを適当に初期化したほうがいいですね(値が爆発しないように)
        v.data = np.ones(v.shape).astype(np.float32) * 4.0

        # print(na,v)
        initializers.append(convert_parameter(v, na.replace('/', '_')))

    dummygraph = helper.make_graph(
        [], "hoge", [], [], initializer=initializers)
    dummygraph.ClearField("name")
    # print(dummygraph)
    onnxmod.graph.MergeFrom(dummygraph)


def run_chainer_model(model, x, out_key):
    # Forward computation
    if isinstance(x, (list, tuple)):
        for i in x:
            assert isinstance(i, (np.ndarray, chainer.Variable))
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

    if isinstance(chainer_out, (list, tuple)):
        chainer_out = [y.array for y in chainer_out]
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


def check_compatibility(model, x, out_key='prob'):
    args = test_args.get_test_args()

    if not MXNET_AVAILABLE:
        raise ImportError('check_compatibility requires MXNet.')

    # さらの状態からonnxのmodをつくる
    onnxmod, input_tensors, output_tensors = chainer2onnx.chainer2onnx(
        model, model.forward)
    checker.check_model(onnxmod)

    with open(args.raw_output, 'wb') as fp:
        fp.write(onnxmod.SerializeToString())

    chainer.config.train = False
    run_chainer_model(model, x, out_key)

    print("parameter initialized")  # これより前のoverflowは気にしなくて良いはず
    # 1回の実行をもとにinitialize
    edit_onnx_protobuf(onnxmod, x, model)
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
        assert len(input_names) == 1

        assert len(output_tensors) == len(chainer_out)
        outputs = []
        for tensor, value in zip(output_tensors, chainer_out):
            outputs.append((tensor.name, value))

        dump_test_inputs_outputs(
            [(input_names[0], x)],
            outputs,
            args.test_data_dir)

    if not args.check:
        return

    fn = 'o.onnx'
    with open(fn, 'wb') as fp:
        fp.write(onnxmod.SerializeToString())

    # onnx_chainer.export(model, x, fn)

    sym, arg, aux = mxnet.contrib.onnx.import_model(fn)

    data_names = [graph_input for graph_input in sym.list_inputs()
                  if graph_input not in arg and graph_input not in aux]
    if len(data_names) > 1:
        data_shapes = [(n, x_.shape) for n, x_ in zip(data_names, x)]
    else:
        data_shapes = [(data_names[0], x.shape)]

    # print(aux)
    # print(data_names)
    print('data shape', data_shapes)
    # import onnx
    # print(onnx.load('o.onnx'))

    mod = mxnet.mod.Module(
        symbol=sym, data_names=data_names, context=mxnet.cpu(),
        label_names=None)
    mod.bind(
        for_training=False, data_shapes=data_shapes,
        label_shapes=None)
    mod.set_params(
        arg_params=arg, aux_params=aux, allow_missing=True,
        allow_extra=True)

    Batch = collections.namedtuple('Batch', ['data'])
    if isinstance(x, (list, tuple)):
        x = [mxnet.nd.array(x_.array) if isinstance(
            x_, chainer.Variable) else mxnet.nd.array(x_) for x_ in x]
    elif isinstance(x, chainer.Variable):
        x = [mxnet.nd.array(x.array)]
    elif isinstance(x, np.ndarray):
        x = [mxnet.nd.array(x)]

    mod.forward(Batch(x))
    mxnet_outs = mod.get_outputs()
    mxnet_out = [y.asnumpy() for y in mxnet_outs]

    print(len(chainer_out), len(mxnet_out))
    print(x[0].shape)
    print('chainershape', chainer_out[0].shape)
    print('mxshape     ', mxnet_out[0].shape)
    # print(x)
    print(chainer_out)
    # print(mxnet_out)

    for cy, my in zip(chainer_out, mxnet_out):
        np.testing.assert_almost_equal(cy, my, decimal=5)

    os.remove(fn)
