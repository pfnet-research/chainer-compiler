"""Yet another ONNX test generator for custom ops and new ops."""


import os

import chainer
import numpy as np
import onnx
from onnx import numpy_helper


F = chainer.functions


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def V(a):
    return chainer.variable.Variable(np.array(a))


def aranges(*shape):
    r = 1
    for d in shape:
        r *= d
    return np.arange(r).reshape(shape).astype(np.float32)


# From onnx/backend/test/case/node/__init__.py
def _extract_value_info(arr, name):
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape)


def expect(node, inputs, outputs, name):
    assert len(node.input) == len(inputs)
    assert len(node.output) == len(outputs)
    inputs_vi = [_extract_value_info(a.array, n)
                 for a, n in zip(inputs, node.input)]
    outputs_vi = [_extract_value_info(a.array, n)
                  for a, n in zip(outputs, node.output)]

    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi)
    model = onnx.helper.make_model(graph, producer_name='backend-test')

    test_dir = os.path.join('out', name)
    test_data_set_dir = os.path.join(test_dir, 'test_data_set_0')
    makedirs(test_data_set_dir)
    with open(os.path.join(test_dir, 'model.onnx'), 'wb') as f:
        f.write(model.SerializeToString())
    for typ, values in [('input', zip(inputs, node.input)),
                        ('output', zip(outputs, node.output))]:
        for i, (value, name) in enumerate(values):
            filename = os.path.join(test_data_set_dir, '%s_%d.pb' % (typ, i))
            tensor = numpy_helper.from_array(value.array, name)
            with open(filename, 'wb') as f:
                f.write(tensor.SerializeToString())


def gen_select_item_test(test_name):
    node = onnx.helper.make_node(
        'OnikuxSelectItem',
        inputs=['input', 'indices'],
        outputs=['output'])

    input = V(aranges(4, 3))
    indices = V([1, 2, 0, 1])
    output = F.select_item(input, indices)
    expect(node, inputs=[input, indices], outputs=[output], name=test_name)


def get_tests():
    return [
        ('extra_test_select_item', gen_select_item_test),
    ]


def main():
    for name, test_fn in get_tests():
        test_fn(name)
    # TODO(hamaji): Stop writing a file to scripts.
    with open('scripts/extra_test_stamp', 'w'): pass


if __name__ == '__main__':
    main()
