import chainer
import contextlib
import os
import pkg_resources
import shutil

import numpy as np
try:
    from onnx_chainer.export import export as onnx_chainer_export
except pkg_resources.DistributionNotFound:
    pass
from onnx import numpy_helper


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
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def _write_tensor_pb(filename, name, value):
    with open(filename, 'wb') as f:
        t = numpy_helper.from_array(value, name)
        f.write(t.SerializeToString())

def create_onnx_test(graph_name, model, inputs, builtins, out_dir):
    # TODO(hamaji): Investigate why we need to set train=False for ResNet50.
    chainer.config.train = False
    makedirs(out_dir)
    with replace_id(model, builtins):
        onnx_chainer_export(model, inputs,
                            filename='%s/model.onnx' % out_dir,
                            graph_name=graph_name)

    onnx_extra_inputs = []
    if hasattr(model, 'extra_inputs'):
        onnx_extra_inputs = model.extra_inputs

    test_data_dir = '%s/test_data_set_0' % out_dir
    makedirs(test_data_dir)
    for i, var in enumerate(list(inputs) + list(onnx_extra_inputs)):
        filename = os.path.join(test_data_dir, 'input_%d.pb' % i)
        _write_tensor_pb(filename, 'Input_%d' % i, var.data)

    chainer.config.train = True
    model.cleargrads()
    result = model(*inputs)
    result.grad = np.ones(result.shape, result.dtype)
    result.backward()

    outputs = [('', result.array)]
    for i, (name, value) in enumerate(outputs):
        filename = os.path.join(test_data_dir, 'output_%d.pb' % i)
        _write_tensor_pb(filename, name, value)
    for name, param in model.namedparams():
        filename = os.path.join(test_data_dir, 'gradient_%d.pb' % i)
        _write_tensor_pb(filename, name, param.grad)
