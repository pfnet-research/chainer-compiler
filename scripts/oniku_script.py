import collections
import os
import shutil

import onnx
from onnx import numpy_helper


# From onnx/backend/test/case/node/__init__.py
def _extract_value_info(arr, name):
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape)


def make_constant_node(name, typ, value):
    tensor = onnx.helper.make_tensor(name + '_val', typ, (), value)
    node = onnx.helper.make_node('Constant', inputs=[], outputs=[name],
                                 value=tensor)
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
            filename = os.path.join(test_data_set_dir, '%s_%d.pb' % (typ, i))
            tensor = numpy_helper.from_array(value, name)
            with open(filename, 'wb') as f:
                f.write(tensor.SerializeToString())


class GraphBuilder(object):
    def __init__(self, graph_name):
        self.graph_name = graph_name
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.ids = collections.defaultdict(int)

    def __getattr__(self, name):
        if not name[0].isupper():
            super(GraphBuilder, self).__getattr__(name)

        def make_node(outputs=None, **kwargs):
            return self.make_node(name, outputs=outputs, **kwargs)

        return make_node

    def make_node(self, name, outputs=None, **kwargs):
        if outputs is None:
            outputs = ['%s_%d' % (name, self.gen_id(name))]
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
        self.gen_name[name] += 1
        return oid

    def input(self, name, value):
        self.inputs.append((name, value))
        return name

    def output(self, name, value):
        self.outputs.append((name, value))
        return name

    def const(self, dtype, value, name=None):
        if name is None:
            name = self.gen_id('const')
        node = make_constant_node(name, dtype, value)
        self.nodes.append(node)
        return name

    def make_graph(self):
        inputs_vi = [_extract_value_info(a, n) for n, a in self.inputs]
        outputs_vi = [_extract_value_info(a, n) for n, a in self.outputs]
        return onnx.helper.make_graph(self.nodes, self.graph_name,
                                      inputs=inputs_vi, outputs=outputs_vi)

    def gen_test(self, graph=None):
        if graph is None:
            graph = self.make_graph()
        gen_test(graph, self.inputs, self.outputs, name=self.graph_name)
