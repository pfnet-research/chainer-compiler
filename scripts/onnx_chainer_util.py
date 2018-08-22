import chainer
import contextlib
import os
import sys

import numpy as np

import onnx_chainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oniku.tools import npz_to_onnx


@contextlib.contextmanager
def replace_id(model, builtins):
    orig_id = builtins.id
    name_map = {}
    param_to_names = {}
    for name, param in model.namedparams():
        param_to_names[id(param)] = name

    def resolve_name(x):
        if orig_id(x) in param_to_names:
            return param_to_names[orig_id(x)]

        param_id = name_map.get(x.name, 0)
        name_map[x.name] = param_id + 1
        name = '%s_%d' % (x.name, param_id) if param_id else x.name
        return name

    def my_id(x):
        if (isinstance(x, chainer.Parameter) or
            (isinstance(x, chainer.Variable) and x.name) or
            (isinstance(x, chainer.variable.VariableNode) and x.name)):
            if hasattr(x, 'onnx_name'):
                return x.onnx_name
            name = resolve_name(x)
            setattr(x, 'onnx_name', name)
            return name
        return orig_id(x)
    builtins.id = my_id
    yield
    builtins.id = orig_id


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def create_onnx_test(graph_name, model, inputs, builtins, out_dir):
    # TODO(hamaji): Investigate why we need to set train=False for ResNet50.
    chainer.config.train = False
    makedirs(out_dir)
    with replace_id(model, builtins):
        onnx_chainer.export(model, inputs,
                            filename='%s/model.onnx' % out_dir,
                            graph_name=graph_name)

    onnx_extra_inputs = []
    if hasattr(model, 'extra_inputs'):
        onnx_extra_inputs = model.extra_inputs

    test_data_dir = '%s/test_data_set_0' % out_dir
    makedirs(test_data_dir)
    for i, var in enumerate(list(inputs) + list(onnx_extra_inputs)):
        with open(os.path.join(test_data_dir, 'input_%d.pb' % i), 'wb') as f:
            t = npz_to_onnx.np_array_to_onnx(var.name, var.data)
            f.write(t.SerializeToString())

    chainer.config.train = True
    model.cleargrads()
    result = model(*inputs)
    result.grad = np.ones(result.shape, result.dtype)
    result.backward()

    outputs = [(result.name, result.array)]
    for name, param in model.namedparams():
        outputs.append(('grad_out@' + name, param.grad))
    for i, (name, value) in enumerate(outputs):
        with open(os.path.join(test_data_dir, 'output_%d.pb' % i), 'wb') as f:
            t = npz_to_onnx.np_array_to_onnx(name, value)
            f.write(t.SerializeToString())
