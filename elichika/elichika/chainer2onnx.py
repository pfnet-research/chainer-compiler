import chainer

import onnx
import onnx.helper as oh
from onnx import TensorProto
from onnx import ModelProto

import elichika.parser.core as core
import elichika.parser.graphs as graphs
import elichika.parser.values as values
import elichika.parser.nodes as nodes
import elichika.parser.functions as functions
import elichika.parser.functions_builtin as functions_builtin
import elichika.parser.values_builtin as values_builtin

import numpy as np
import collections

def size2d(x):
    if isinstance(x, collections.Iterable):
        return x
    return (x, x)

assigned_names = []
node2onnx_parameter = {}
value2onnx_parameter = {}

class NodeONNXParameter:
    def __init__(self, onnx_name, value):
        self.onnx_name = onnx_name
        self.original_value = value

class ValueONNXParameter:
    def __init__(self, onnx_name, value):
        self.onnx_name = onnx_name
        self.original_value = value

def onnx_name(value):
    if isinstance(value, values.Value):
        return value2onnx_parameter[value].onnx_name
    if isinstance(value, nodes.Node):
        return node2onnx_parameter[value].onnx_name

def generate_onnx_value_name(value : 'values.Value', none_name = ''):
    base_name = ''

    if value.generator != None:
        base_name = value.name + '_' + str(value.generator.lineprop)
    base_name = value.name

    if base_name == '':
        base_name = none_name

    ind = 0
    name = base_name

    if name == '':
        name = 'noname'

    while (name in assigned_names):
        ind+=1
        name = base_name + '_' + str(ind)

    assigned_names.append(name)
    return name

def generate_onnx_node_name(node : 'nodes.Node'):
    base_name = str(node)

    ind = 0
    name = base_name
    while (name in assigned_names):
        ind+=1
        name = base_name + '_' + str(ind)

    assigned_names.append(name)
    return name



def assign_onnx_name_to_value(value : 'values.Value', none_name = ''):
    if not value in value2onnx_parameter:
        value2onnx_parameter[value] = ValueONNXParameter(generate_onnx_value_name(value, none_name), value)

    if isinstance(value, values.TupleValue):
        tupleValue = value # type : values.TupleValue
        for value_ in tupleValue.values:
            assign_onnx_name_to_value(value_.get_value(), value2onnx_parameter[tupleValue].onnx_name)


def assign_onnx_name(graph : 'graphs.Graph'):

    for v in graph.input_values:
        assign_onnx_name_to_value(v)

    for v in graph.output_values:
        assign_onnx_name_to_value(v)

    for node in graph.nodes:
        for input in node.inputs:
            assign_onnx_name_to_value(input)

        for output in node.outputs:
            assign_onnx_name_to_value(output)

        if not node in node2onnx_parameter:
            node2onnx_parameter[node] = NodeONNXParameter(generate_onnx_node_name(node), node)

        for subgraph in node.subgraphs:
            assign_onnx_name(subgraph)

def preprocess(graph : 'graphs.Graph'):

    replacing = {}
    for value in graph.output_values:
        if value in graph.input_values:
            copied_value = functions.generate_copied_value(value)
            replacing[value] = copied_value
            node = nodes.NodeCopy(value)
            node.set_outputs([copied_value])
            graph.add_node(node)

    for i in range(len(graph.output_values)):
        if graph.output_values[i] in replacing.keys():
            graph.output_values[i] = replacing[graph.output_values[i]]

    for node in graph.nodes:
        for subgraph in node.subgraphs:
            preprocess(subgraph)

def convert_onnx_chainer_linear(onnx_graph : 'ONNXGraph', node : 'nodes.Node'):
    chainer_inst = node.func.owner.inst # type: chainer.links.Linear
    onnx_name = node2onnx_parameter[node].onnx_name

    x = onnx_graph.tensors[value2onnx_parameter[node.inputs[0]].onnx_name]
    o = onnx_graph.tensors[value2onnx_parameter[node.outputs[0]].onnx_name]

    if chainer_inst.W.data is None:
        print("W is unknown. Please infer this model.")

    w = onnx_graph.new_tensor_with_np(chainer_inst.W.data, onnx_name + '/W')

    x_shape = onnx_graph.new_empty_tensor(['TODO'], np.float32, onnx_name + '/x_shape')
    batch_size_1 = onnx_graph.new_empty_tensor(['TODO'], np.float32, onnx_name + '/batch_size_1')
    batch_size_2 = onnx_graph.new_empty_tensor(['TODO'], np.float32, onnx_name + '/batch_size_2')
    mat_shape = onnx_graph.new_empty_tensor(['TODO'], np.float32, onnx_name + '/mat_shape')
    x_reshape = onnx_graph.new_empty_tensor(['TODO'], np.float32, onnx_name + '/x_reshape')

    onnx_graph.add_node(
        'Shape',
        [x.name],
        [x_shape.name],
        str(node.lineprop))

    onnx_graph.add_node(
        'Gather',
        [x_shape.name, onnx_graph.new_tensor_with_np(np.array(0, dtype=np.int64), onnx_name + '/Zero').name],
        [batch_size_1.name],
        str(node.lineprop))

    onnx_graph.add_node(
        'Unsqueeze',
        [batch_size_1.name],
        [batch_size_2.name],
        str(node.lineprop),
        axes=[0])

    onnx_graph.add_node(
        'Concat',
        [batch_size_2.name, onnx_graph.new_tensor_with_np(np.array([-1], dtype=np.int64), onnx_name + '/Minus1').name],
        [mat_shape.name],
        str(node.lineprop),
        axis=0)

    onnx_graph.add_node(
        'Reshape',
        [x.name, mat_shape.name],
        [x_reshape.name],
        str(node.lineprop))

    x = x_reshape

    if chainer_inst.b is not None:
        b = onnx_graph.new_tensor_with_np(chainer_inst.b.data, onnx_name + '/B')

        onnx_graph.add_node(
            'Gemm',
            [x.name, w.name, b.name],
            [o.name],
            str(node.lineprop),
            transA=0,
            transB=1)
    else:
        temp = onnx_graph.new_empty_tensor(['TODO'], np.float32, onnx_name + '/Temp')
        onnx_graph.add_node(
            'Transpose',
            [w.name],
            [temp.name],
            str(node.lineprop),
            perm=[1, 0])

        onnx_graph.add_node(
            'MatMul',
            [x.name, temp.name],
            [o.name],
            str(node.lineprop))

def convert_onnx_chainer_convolution2d(onnx_graph : 'ONNXGraph', node : 'nodes.Node'):
    chainer_inst = node.func.owner.inst # type: chainer.links.Convolution2D
    onnx_name = node2onnx_parameter[node].onnx_name

    ksize = size2d(chainer_inst.ksize)
    stride = size2d(chainer_inst.stride)
    ps = size2d(chainer_inst.pad)
    pads = ps + ps

    x = onnx_graph.tensors[value2onnx_parameter[node.inputs[0]].onnx_name]
    o = onnx_graph.tensors[value2onnx_parameter[node.outputs[0]].onnx_name]
    w = onnx_graph.new_tensor_with_np(chainer_inst.W.data, onnx_name + '/W')
    b = None

    if chainer_inst.b is not None:
        b = onnx_graph.new_tensor_with_np(chainer_inst.b.data, onnx_name + '/b')

    onnx_graph.add_node(
        'Conv',
        [x.name, w.name] + ([] if b is None else [b.name]),
        [o.name],
        str(node.lineprop),
        kernel_shape=ksize,
        pads=pads,
        strides=stride)


class ONNXInitrializer:
    def __init__(self):
        self.node = None
        self.name = NameError
        self.dt = 0
        self.shape = ()

class ONNXGraph:
    def __init__(self, generator : 'ONNXGenerator', parent : 'ONNXGraph'):
        self.generator = generator
        self.parent = parent
        self.nodes = []
        self.input_tensor = []
        self.output_tensor = []
        self.tensors = {}
        self.onnx_tensors = {}

    def try_get_attribute(self, value):
        if isinstance(value, values.NumberValue):
            value_ = value  # type: values.NumberValue
            return value_.internal_value

        if isinstance(value, values.BoolValue):
            value_ = value  # type: values.BoolValue
            return value_.internal_value

        # error
        print("Cannot convert a value into an attribute")
        return -1

    def new_empty_tensor(self, dims, dtype, name):
        '''
        generate a tensor for connecting between nodes
        '''
        dt = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        tensor = oh.make_tensor_value_info(name, dt, dims)
        self.tensors[name] = tensor
        return tensor

    def new_empty_tensor_with_value(self, value):
        '''
        generate a tensor with Value to indicate shape
        it is for inputting and outputting
        '''
        if isinstance(value, values.TensorValue) and len(value.shape) > 0:
            shape = list(value.shape)
            shape = [x if x != -1 else 'Undefined' for x in shape]
            return self.new_empty_tensor(shape, np.float32, value2onnx_parameter[value].onnx_name)

        return self.new_empty_tensor(['Undefined'], np.float32, value2onnx_parameter[value].onnx_name)

    def new_tensor_with_np(self, ndarray_, name):
        '''
        generate a tensor which contains np data
        it is for constant input
        '''
        dt = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(ndarray_.dtype)]
        tensor = oh.make_tensor(name, dt, ndarray_.shape, ndarray_.flat)
        initializer = ONNXInitrializer()
        initializer.name = name
        initializer.node = tensor
        initializer.dt = dt
        initializer.shape = ndarray_.shape

        assert(not (name in self.generator.initializers.keys()))
        self.generator.initializers[name] = initializer

        return tensor

    def new_tensor_with_value(self, value):
        '''
        generate a tensor which value
        it is for constant input
        '''
        name = value2onnx_parameter[value].onnx_name

        if isinstance(value, values.NumberValue):
            arr = np.array(value.internal_value)
            return self.new_tensor_with_np(arr, name)

        if isinstance(value, values.BoolValue):
            arr = np.array(value.internal_value)
            return self.new_tensor_with_np(arr, name)

        if isinstance(value, values.NoneValue):
            arr = np.array(False)
            return self.new_tensor_with_np(arr, name)


        print('Warning : Found uknown type {} in new_tensor_with_value. Float is stored.'.format(type(value)))
        arr = np.array(0.0, dtype=np.float32)
        return self.new_tensor_with_np(arr, name)

    def add_node(self, optype, inputs, outputs, name, **kwargs):
        # check types
        assert(len([i for i in inputs if not isinstance(i, str)]) == 0)
        assert(len([i for i in outputs if not isinstance(i, str)]) == 0)

        node = oh.make_node(optype, inputs, outputs, name, **kwargs)
        self.nodes.append(node)

    def try_get_tensor(self, onnx_name : 'str'):
        if onnx_name in self.tensors.keys():
            return self.tensors[onnx_name]

        #if self.parent is not None:
        #    return self.parent.try_get_tensor(onnx_name)

        return None

    def set_input(self, input):
        self.input_tensor = []

        for input_ in input:
            onnx_name = value2onnx_parameter[input_].onnx_name
            value = self.try_get_tensor(onnx_name)
            assert(value is not None)
            self.input_tensor.append(value)

    def set_output(self, output):
        self.output_tensor = [self.tensors[value2onnx_parameter[x].onnx_name] for x in output]

    def generate_graph(self, name : 'str', isMain = False):

        input_tensor_and_initializer = self.input_tensor.copy()
        initializers = []

        # add initializers
        if isMain:
            for v in self.generator.initializers.values():
                if v.node in self.input_tensor:
                    continue
                if v.node in self.output_tensor:
                    continue

                initializers.append(v.node)

                tensor = oh.make_tensor_value_info(v.name, v.dt, v.shape)
                input_tensor_and_initializer.append(tensor)

        return oh.make_graph(self.nodes, name, input_tensor_and_initializer, self.output_tensor, initializer=initializers)

class ONNXGenerator:
    def __init__(self):
        self.onnx_graphs = []
        self.initializers = {}

    def generate_graph(self, inputs, outputs, graph : 'graphs.Graph', parent : 'ONNXGraph', isMain = False):
        onnx_graph = ONNXGraph(self, parent)

        def generate_input_tensors(inputs_):
            for input in inputs_:
                if not (value2onnx_parameter[input].onnx_name in onnx_graph.onnx_tensors.keys()):

                    if input.generator is None and not (input in inputs):
                        tensor = onnx_graph.new_tensor_with_value(input)
                        onnx_graph.onnx_tensors[value2onnx_parameter[input].onnx_name] = tensor
                    else:
                        tensor = onnx_graph.new_empty_tensor_with_value(input)
                        onnx_graph.onnx_tensors[value2onnx_parameter[input].onnx_name] = tensor


        def generate_output_tensors(outputs_):
            for output in outputs_:
                if not (value2onnx_parameter[output].onnx_name in onnx_graph.onnx_tensors.keys()):
                    tensor = onnx_graph.new_empty_tensor_with_value(output)
                    onnx_graph.onnx_tensors[value2onnx_parameter[output].onnx_name] = tensor

        generate_input_tensors(inputs)

        for node in graph.nodes:
            generate_input_tensors(node.inputs)
            generate_output_tensors(node.outputs)

        generate_output_tensors(outputs)

        for node in graph.nodes:
            if isinstance(node, nodes.NodeCopy):
                node_ = node # type: nodes.Copy
                onnx_node = oh.make_node(
                    'Identity',
                    [value2onnx_parameter[node_.value].onnx_name],
                    [value2onnx_parameter[node.outputs[0]].onnx_name])

                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeNonVolatileAssign):
                node_ = node # type: nodes.NodeNonVolatileAssign
                onnx_node = oh.make_node(
                    'Identity',
                    [value2onnx_parameter[node_.target_value].onnx_name],
                    [value2onnx_parameter[node_.value].onnx_name])

                onnx_graph.nodes.append(onnx_node)


            if isinstance(node, nodes.NodeAugAssign):
                node_ = node # type: nodes.AugAssign
                binops = {}
                binops[nodes.BinOpType.Add] = 'Add'
                binops[nodes.BinOpType.Sub] = 'Sub'
                binops[nodes.BinOpType.Unknown] = 'Add'

                # TODO: fix for reference types

                onnx_node = oh.make_node(
                    binops[node_.binop],
                    [value2onnx_parameter[node_.target].onnx_name,
                    value2onnx_parameter[node_.value].onnx_name],
                    [value2onnx_parameter[node.target].onnx_name])
                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeValueAugAssign):
                node_ = node # type: nodes.ValueAugAssign
                binops = {}
                binops[nodes.BinOpType.Add] = 'Add'
                binops[nodes.BinOpType.Sub] = 'Sub'
                binops[nodes.BinOpType.Unknown] = 'Add'

                onnx_node = oh.make_node(
                    binops[node_.binop],
                    [value2onnx_parameter[node_.target].onnx_name,
                    value2onnx_parameter[node_.value].onnx_name],
                    [value2onnx_parameter[node_.outputs[0]].onnx_name])
                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeBinOp):
                node_ = node # type: nodes.NodeBinOp
                binops = {}
                binops[nodes.BinOpType.Add] = 'Add'
                binops[nodes.BinOpType.Sub] = 'Sub'
                binops[nodes.BinOpType.Unknown] = 'Add'

                onnx_node = oh.make_node(binops[node_.binop], [value2onnx_parameter[node_.left].onnx_name, value2onnx_parameter[node_.right].onnx_name], [value2onnx_parameter[node.outputs[0]].onnx_name])
                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeUnaryOp):
                node_ = node # type: nodes.NodeUnaryOp

                if node_.unaryop == nodes.UnaryOpType.UAdd:
                    zero_ = onnx_graph.new_tensor_with_np(np.array(0, dtype=np.float), node2onnx_parameter[node_].onnx_name + '/Zero')
                    onnx_node = oh.make_node(
                        'Add',
                        [zero_.name, value2onnx_parameter[node_.right].onnx_name],
                        [value2onnx_parameter[node.outputs[0]].onnx_name])
                    onnx_graph.nodes.append(onnx_node)

                if node_.unaryop == nodes.UnaryOpType.USub:
                    zero_ = onnx_graph.new_tensor_with_np(np.array(0, dtype=np.float), node2onnx_parameter[node_].onnx_name + '/Zero')
                    onnx_node = oh.make_node(
                        'Sub',
                        [zero_.name, value2onnx_parameter[node_.right].onnx_name],
                        [value2onnx_parameter[node.outputs[0]].onnx_name])
                    onnx_graph.nodes.append(onnx_node)

                if node_.unaryop == nodes.UnaryOpType.Not:
                    onnx_node = oh.make_node(
                        'Not',
                        [value2onnx_parameter[node_.right].onnx_name],
                        [value2onnx_parameter[node.outputs[0]].onnx_name])
                    onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeCompare):
                node_ = node # type: nodes.NodeCompare

                op_str = None
                op_not = False

                if node_.compare == nodes.CompareType.Eq:
                    op_str = 'Equal'
                if node_.compare == nodes.CompareType.NotEq:
                    op_str = 'Equal'
                    op_not = True
                if node_.compare == nodes.CompareType.Gt:
                    op_str = 'Greater'
                if node_.compare == nodes.CompareType.GtE:
                    op_str = 'Less'
                    op_not = True
                if node_.compare == nodes.CompareType.Lt:
                    op_str = 'Less'
                if node_.compare == nodes.CompareType.LtE:
                    op_str = 'Greater'
                    op_not = True
                if node_.compare == nodes.CompareType.Is:
                    op_str = 'ChainerGenericIs'
                if node_.compare == nodes.CompareType.IsNot:
                    op_str = 'ChainerGenericIs'
                    op_not = True

                if op_not:
                    op_not_temp = onnx_graph.new_empty_tensor(['TODO'], np.bool, value2onnx_parameter[node.outputs[0]].onnx_name + '/NotTemp')
                    onnx_node1 = oh.make_node(op_str, [value2onnx_parameter[node_.left].onnx_name, value2onnx_parameter[node_.right].onnx_name], [op_not_temp.name])
                    onnx_node2 = oh.make_node('Not', [op_not_temp.name], [value2onnx_parameter[node.outputs[0]].onnx_name])
                    onnx_graph.nodes.append(onnx_node1)
                    onnx_graph.nodes.append(onnx_node2)
                else:
                    onnx_node = oh.make_node(op_str, [value2onnx_parameter[node_.left].onnx_name, value2onnx_parameter[node_.right].onnx_name], [value2onnx_parameter[node.outputs[0]].onnx_name])
                    onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeGetItem):
                node_ = node # type: nodes.NodeGetItem
                onnx_node = oh.make_node(
                    'ChainerGenericGetItem',
                    [value2onnx_parameter[node_.target].onnx_name, value2onnx_parameter[node_.index].onnx_name],
                    [value2onnx_parameter[node.outputs[0]].onnx_name])
                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeSlice):
                node_ = node # type: nodes.NodeSlice
                onnx_node = oh.make_node(
                    'ChainerSequenceGetSlice',
                    [value2onnx_parameter[node_.target].onnx_name, value2onnx_parameter[node_.left].onnx_name, value2onnx_parameter[node_.right].onnx_name],
                    [value2onnx_parameter[node.outputs[0]].onnx_name])
                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeCall):

                if isinstance(node.func, functions_builtin.ReluFunction):
                    # relu
                    onnx_node = oh.make_node("Relu", [value2onnx_parameter[node.inputs[0]].onnx_name], [value2onnx_parameter[node.outputs[0]].onnx_name])
                    onnx_graph.nodes.append(onnx_node)

                if isinstance(node.func, functions_builtin.SoftmaxFunction):
                    # softmax
                    onnx_node = oh.make_node(
                        "Softmax",
                        [value2onnx_parameter[node.inputs[0]].onnx_name],
                        [value2onnx_parameter[node.outputs[0]].onnx_name],
                        str(node.lineprop),
                        axis = onnx_graph.try_get_attribute(node.inputs[1]))

                    onnx_graph.nodes.append(onnx_node)

                if isinstance(node.func, values_builtin.ChainerLinkFunction):
                    original_inst = node.func.owner.inst

                    if isinstance(original_inst, chainer.links.Linear):
                        convert_onnx_chainer_linear(onnx_graph, node)

                    if isinstance(original_inst, chainer.links.Convolution2D):
                        convert_onnx_chainer_convolution2d(onnx_graph, node)

            if isinstance(node, nodes.NodeIf):
                node_ = node # type: nodes.NodeIf

                true_graph = self.generate_graph(node_.true_graph.input_values, node_.true_graph.output_values, node_.true_graph, onnx_graph)
                false_graph = self.generate_graph(node_.false_graph.input_values, node_.false_graph.output_values, node_.false_graph, onnx_graph)

                onnx_node = oh.make_node(
                    'If',
                    [value2onnx_parameter[node_.cond].onnx_name] + [value2onnx_parameter[x].onnx_name for x in node.input_values],
                    [value2onnx_parameter[x].onnx_name for x in node.outputs],
                    then_branch=true_graph,
                    else_branch=false_graph)

                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeFor):
                node_ = node # type: nodes.NodeFor

                # get length of sequence
                op_len = onnx_graph.new_empty_tensor(['TODO'], np.int, value2onnx_parameter[node_.iter_value].onnx_name + '/Len')

                onnx_node = oh.make_node(
                    'ChainerGenericLen',
                    [value2onnx_parameter[node_.iter_value].onnx_name],
                    [op_len.name])
                onnx_graph.nodes.append(onnx_node)

                body_graph = self.generate_graph(node_.body_graph.input_values, node_.body_graph.output_values, node_.body_graph, onnx_graph)

                # for
                onnx_node = oh.make_node(
                    'Loop',
                    [op_len.name] + [""] + [value2onnx_parameter[node_.iter_value].onnx_name] + [value2onnx_parameter[x].onnx_name for x in node.input_values],
                    [value2onnx_parameter[x].onnx_name for x in node.outputs],
                    body=body_graph)
                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeForGenerator):
                node_ = node # type: nodes.NodeForGenerator

                # get value from sequence with index
                onnx_node = oh.make_node(
                    'ChainerSequenceLookup',
                    [value2onnx_parameter[node_.iter_value].onnx_name, value2onnx_parameter[node_.counter_value].onnx_name],
                    [value2onnx_parameter[node_.outputs[0]].onnx_name])
                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeListcomp):
                node_ = node # type: nodes.NodeListcomp

                # get length of sequence
                op_len = onnx_graph.new_empty_tensor(['TODO'], np.int, value2onnx_parameter[node_.iter_value].onnx_name + '/Len')

                onnx_node = oh.make_node(
                    'ChainerGenericLen',
                    [value2onnx_parameter[node_.iter_value].onnx_name],
                    [op_len.name])
                onnx_graph.nodes.append(onnx_node)

                body_graph = self.generate_graph(node_.body_graph.input_values, node_.body_graph.output_values, node_.body_graph, onnx_graph)

                onnx_node = oh.make_node(
                    'Loop',
                    [op_len.name] + [""] + [value2onnx_parameter[node_.iter_value].onnx_name] + [value2onnx_parameter[x].onnx_name for x in node.input_values],
                    [value2onnx_parameter[x].onnx_name for x in node.outputs],
                    body=body_graph)

                onnx_graph.nodes.append(onnx_node)

            if isinstance(node, nodes.NodeGenerate):
                node_ = node # type: nodes.NodeGenerate
                if node_.classtype == 'range':
                    onnx_node = oh.make_node(
                        "ChainerSequenceRange",
                        [value2onnx_parameter[input].onnx_name for input in node.inputs],
                        [value2onnx_parameter[node.outputs[0]].onnx_name],
                        str(node.lineprop))

                    onnx_graph.nodes.append(onnx_node)

                if node_.classtype == 'List':
                    last_name = value2onnx_parameter[node.outputs[0]].onnx_name
                    name = last_name
                    count = 0
                    if(len(node_.args) > 0):
                        name += '_gen_' + str(count)

                    onnx_node = oh.make_node(
                        "ChainerSequenceCreate",
                        [],
                        [name],
                        str(node.lineprop))
                    onnx_graph.nodes.append(onnx_node)

                    for i in range(len(node_.args)):
                        next_name = last_name + '_gen_' + str(count + 1)

                        if i == len(node_.args) - 1:
                            next_name = last_name

                        onnx_node = oh.make_node(
                            "ChainerSequenceAppend",
                            [name, value2onnx_parameter[node.args[i]].onnx_name],
                            [next_name],
                            str(node.lineprop))
                        onnx_graph.nodes.append(onnx_node)
                        name = next_name
                        count += 1

        onnx_graph.set_input(inputs)
        onnx_graph.set_output(outputs)

        return onnx_graph.generate_graph(graph.name, isMain=isMain)

    def generate_model(self, inputs, outputs, graph)-> 'ModelProto':
        # assign names
        assigned_names.clear()
        node2onnx_parameter.clear()
        value2onnx_parameter.clear()

        assign_onnx_name(graph)

        graph_ = self.generate_graph(inputs, outputs, graph, None, True)
        model = oh.make_model(graph_, producer_name="elichika", producer_version="0.1")
        return model

class ONNXModel:
    def __init__(self):
        self.model = None
        self.inputs = []
        self.outputs = []

def compile_model(model, inputs) -> 'ONNXModel':
    inputs_, outputs_, graph_ = core.convert_model(model, inputs)

    if graph_ is None:
        return None

    preprocess(graph_)

    generator = ONNXGenerator()
    model = generator.generate_model(graph_.input_values, graph_.output_values, graph_)

    # check inputs


    onnx_model = ONNXModel()
    onnx_model.model = model
    onnx_model.inputs = graph_.input_values
    onnx_model.outputs = graph_.output_values
    return onnx_model

def save_model(path : 'str', model : 'ModelProto'):
    with open(path, "wb") as f:
        f.write(model.SerializeToString())

def save_model_as_text(path : 'str', model : 'ModelProto'):
    with open(path, "w") as f:
        print(model, file=f)
