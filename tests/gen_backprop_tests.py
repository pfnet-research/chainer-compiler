import os
import sys

import chainer
import numpy as np
from onnx import onnx_pb
import onnx_chainer


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


class AnyModel(chainer.Chain):
    def __init__(self, fn, params):
        super(AnyModel, self).__init__()
        with self.init_scope():
            for name, value in params.items():
                setattr(self, name, chainer.Parameter(value))
        self.fn = fn

    def __call__(self, *args):
        return self.fn(self, *args)


def create_backprop_test(test_name, fn, **kwargs):
    test_dir = 'out/backprop_test_%s' % test_name
    test_model_path = os.path.join(test_dir, 'model.onnx')
    test_data_dir = os.path.join(test_dir, 'test_data_set_0')
    makedirs(test_data_dir)

    params = {}
    for name, value in kwargs.items():
        params[name] = np.array(value, np.float32)
    model = AnyModel(fn, params)

    onnx_chainer.export(model, (), filename=test_model_path)

    model.cleargrads()
    result = model()
    result.backward()

    param_names = []
    output_names = []
    xmodel = onnx_pb.ModelProto()
    with open(test_model_path, 'rb') as f:
        xmodel.ParseFromString(f.read())
        for param in xmodel.graph.initializer:
            assert param.name
            param_names.append(param.name)
        for output in xmodel.graph.output:
            assert output.name
            output_names.append(output.name)

    assert len(output_names) == 1
    assert len(param_names) == len(params)

    outputs = [(output_names[0], result)]
    for xname, name in zip(param_names, sorted(params)):
        value = getattr(model, name).grad
        outputs.append(('grad_out@' + xname, value))

    for i, (xname, value) in enumerate(outputs):
        tensor = onnx_pb.TensorProto(name=xname,
                                     dims=value.shape,
                                     data_type=onnx_pb.TensorProto.FLOAT)
        tensor.float_data.extend(np.array(value.data).flat)
        with open(os.path.join(test_data_dir, 'output_%d.pb' % i), 'wb') as f:
            f.write(tensor.SerializeToString())


def main():
    create_backprop_test('add', lambda m: m.a + m.b, a=[3], b=[7])
    create_backprop_test('mul', lambda m: m.a * m.b, a=[3], b=[7])


if __name__ == '__main__':
    main()
