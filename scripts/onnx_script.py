import collections
import os
import shutil

import chainer
import numpy as np
import onnx
from onnx import numpy_helper


# From onnx/backend/test/case/node/__init__.py
def _extract_value_info(arr, name):
    if isinstance(arr, list):
        assert arr
        assert not isinstance(arr[0], list)
        value_info_proto = onnx.ValueInfoProto()
        value_info_proto.name = name
        sequence_type_proto = value_info_proto.type.sequence_type
        nested = _extract_value_info(arr[0], name)
        tensor_type = sequence_type_proto.elem_type.tensor_type
        tensor_type.CopyFrom(nested.type.tensor_type)
        return value_info_proto
    else:
        return onnx.helper.make_tensor_value_info(
            name=name,
            elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
            shape=arr.shape)


def make_constant_node(name, typ, value):
    value = np.array(value)
    tensor = onnx.helper.make_tensor(name + '_val', typ,
                                     value.shape, value.flat)
    node = onnx.helper.make_node('Constant', inputs=[], outputs=[name],
                                 value=tensor)
    return node


def make_constant_sequence_node(name, typ, values):
    tensor_values = []
    for i, value in enumerate(values):
        value = np.array(value)
        tensor = onnx.helper.make_tensor('%s_%d' % (name, i), typ,
                                         value.shape, value.flat)
        tensor_values.append(tensor)
    node = onnx.helper.make_node('ChainerSequenceConstants',
                                 inputs=[], outputs=[name],
                                 tensor_values=tensor_values)
    return node


def gen_test(graph, inputs, outputs, name):
    model = onnx.helper.make_model(graph, producer_name='backend-test')

    test_dir = os.path.join('out', name)
    test_data_set_dir = os.path.join(test_dir, 'test_data_set_0')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_data_set_dir)
    with open(os.path.join(test_dir, 'model.onnx'), 'wb') as f:
        f.write(model.SerializeToString())
    for typ, values in [('input', inputs), ('output', outputs)]:
        for i, (name, value) in enumerate(values):
            if isinstance(value, list):
                assert value
                digits = len(str(len(value)))
                for j, v in enumerate(value):
                    filename = os.path.join(
                        test_data_set_dir,
                        '%s_%d_%s.pb' % (typ, i, str(j).zfill(digits)))
                    tensor = numpy_helper.from_array(v, name)
                    with open(filename, 'wb') as f:
                        f.write(tensor.SerializeToString())
            else:
                filename = os.path.join(test_data_set_dir,
                                        '%s_%d.pb' % (typ, i))
                tensor = numpy_helper.from_array(value, name)
                with open(filename, 'wb') as f:
                    f.write(tensor.SerializeToString())


class Seq(object):
    """Wraps a Python list to clearly identify sequence inputs/outputs."""

    def __init__(self, l):
        assert isinstance(l, list)
        self.list = l


def _array(value, dtype=None):
    if dtype is None and isinstance(value, float):
        dtype = np.float32
    return np.array(value, dtype=dtype)


def _validate_inout(value):
    if isinstance(value, Seq):
        return list(map(np.array, value.list))
    else:
        return _array(value)


class GraphBuilder(object):
    # Shared among GraphBuilder instances so value names will be unique.
    ids = collections.defaultdict(int)

    def __init__(self, graph_name):
        self.graph_name = graph_name
        self.nodes = []
        self.inputs = []
        self.params = []
        self.outputs = []
        self.gradients = []
        self.ids = GraphBuilder.ids

    def __getattr__(self, name):
        if not name[0].isupper():
            raise AttributeError('Unknown attribute: %s' % name)

        def make_node(inputs=[], outputs=None, **kwargs):
            return self.make_node(name, inputs=inputs, outputs=outputs,
                                  **kwargs)

        return make_node

    def make_node(self, name, outputs=None, **kwargs):
        if outputs is None:
            outputs = [self.gen_id(name)]
        elif isinstance(outputs, str):
            outputs = [outputs]
        node = onnx.helper.make_node(name, outputs=outputs, **kwargs)
        self.nodes.append(node)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def gen_id(self, name):
        oid = self.ids[name]
        self.ids[name] += 1
        return '%s_%d' % (name, oid)

    def input(self, name, value):
        self.inputs.append((name, _validate_inout(value)))
        return name

    def param(self, name, value):
        self.params.append((name, _array(value)))
        return name

    def output(self, name, value):
        if isinstance(value, chainer.variable.Variable):
            value = value.array
        self.outputs.append((name, _validate_inout(value)))
        return name

    def gradient(self, name, value):
        self.gradients.append(('grad_out@' + name, _validate_inout(value)))

    def const(self, value, dtype=None, name=None):
        value = _array(value, dtype=dtype)
        if not isinstance(dtype, int):
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]

        if name is None:
            name = self.gen_id('const')
        node = make_constant_node(name, dtype, value)
        self.nodes.append(node)
        return name

    def const_seq(self, values, dtype=None, name=None):
        example = _array(values[0], dtype=dtype)
        if not isinstance(dtype, int):
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[example.dtype]

        if name is None:
            name = self.gen_id('const_seq')
        node = make_constant_sequence_node(name, dtype, values)
        self.nodes.append(node)
        return name

    def make_graph(self):
        inputs_vi = [_extract_value_info(a, n)
                     for n, a in self.inputs + self.params]
        outputs_vi = [_extract_value_info(a, n) for n, a in self.outputs]
        initializer = []
        for name, value in self.params:
            typ = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]
            tensor = onnx.helper.make_tensor(
                name, typ, value.shape, value.flat)
            initializer.append(tensor)
        graph = onnx.helper.make_graph(self.nodes, self.graph_name,
                                       inputs=inputs_vi, outputs=outputs_vi,
                                       initializer=initializer)
        return graph

    def gen_test(self, graph=None):
        if graph is None:
            graph = self.make_graph()
        outputs = self.outputs + self.gradients
        gen_test(graph, self.inputs, outputs, name=self.graph_name)
