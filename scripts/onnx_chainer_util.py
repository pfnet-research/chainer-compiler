import chainer
import contextlib


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
            isinstance(x, chainer.Variable) and x.name):
            if hasattr(x, 'onnx_name'):
                return x.onnx_name
            name = resolve_name(x)
            setattr(x, 'onnx_name', name)
            return name
        return orig_id(x)
    builtins.id = my_id
    yield
    builtins.id = orig_id
