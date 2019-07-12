import chainer
import chainer.functions as F
import chainer.links as L
import inspect
from enum import Enum

from chainer_compiler.elichika.parser import core
from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import utils


class BinOpType(Enum):
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    FloorDiv = 4,
    Mod = 5,
    Unknown = 255,


class UnaryOpType(Enum):
    UAdd = 0,
    USub = 1,
    Not = 2,
    Unknown = 255,

class MultiaryOpType(Enum):
    And = 0,
    Or = 1,
    Unknown = 255,

class CompareType(Enum):
    Eq = 0,
    NotEq = 1,
    Gt = 2,
    GtE = 3,
    Lt = 4,
    LtE = 5,
    Is = 6,
    IsNot = 7,
    unknown = 255,

def make_attribute(value):
    if isinstance(value, list):
        for i in range(len(value)):
            value[i] = make_attribute(value[i])

    if isinstance(value, functions.FunctionArgValueInput):
        converted = {}

        ret = functions.FunctionArgValueInput()

        for v in value.inputs:
            converted_v = make_attribute(v)
            ret.inputs.append(converted_v)
            converted[v] = converted_v

        keywords_ = {}
        for k,v in value.keywords.items():
            if v in converted.keys():
                keywords_[k] = converted[v]
            else:
                keywords_[k] = make_attribute(v)
        ret.keywords = keywords_
        return ret

    if isinstance(value, values.TupleValue) and value.internal_value is not None:
        vs = []
        for v in value.internal_value:
            if isinstance(v, values.ValueRef):
                v = v.get_value()
            vs.append(v)

        ret = values.TupleValue(vs)
        ret.name = value.name
        ret.generator = value.generator

        return ret

    if isinstance(value, values.ValueRef):
        return value.get_value()

    return value

class Node:
    def __init__(self, line):
        self.inputs = []
        self.outputs = []
        self.subgraphs = []
        self.lineprop = utils.LineProperty()

        if isinstance(line, int):
            self.lineprop.lineno = line
        else:
            self.lineprop = line

        return

    def append_inputs(self, input):
        assert(input is not None)
        self.inputs.append(input)

    def extend_inputs(self, inputs):
        for input in inputs:
            assert(input is not None)
        self.inputs.extend(inputs)

    def set_outputs(self, outputs):
        for output in outputs:
            assert(output is not None)

        self.outputs = outputs

        for output in self.outputs:
            assert(output.generator is None)
            output.generator = self

class NodeInvalid(Node):
    def __init__(self, line=-1):
        super().__init__(line)

    def __str__(self):
        return 'Invalid({})'.format(self.lineprop)

class NodeInput(Node):
    def __init__(self, tag = '', line=-1):
        super().__init__(line)
        self.tag = tag

    def __str__(self):
        return 'Input({})'.format(self.tag)

class NodeCopy(Node):
    def __init__(self, value: 'values.Value', line=-1):
        super().__init__(line)
        self.value = value
        self.append_inputs(value)

    def __str__(self):
        return 'Copy({})'.format(self.lineprop)


class NodeNonVolatileAssign(Node):
    def __init__(self, target_value: 'values.Value', value: 'values.Value', line=-1):
        super().__init__(line)
        self.target_value = target_value
        self.value = value
        self.append_inputs(target_value)
        self.append_inputs(value)

    def __str__(self):
        return 'NodeNonVolatileAssign({})'.format(self.lineprop)


class NodeAssign(Node):
    def __init__(self, attr: 'values.Attribute', obj: 'values.ValueRef', line=-1):
        assert(isinstance(obj, values.ValueRef))
        super().__init__(line)

        self.targets = []
        self.objects = []

        self.targets.append(attr)
        self.objects.append(obj)

    def __str__(self):
        return 'Assign({})'.format(self.lineprop)


class NodeAugAssign(Node):
    def __init__(self, target: 'values.Value', value: 'values.Value', binop: 'BinOp', line=-1):
        super().__init__(line)
        self.target = target
        self.value = value
        self.binop = binop

        self.append_inputs(target)
        self.append_inputs(value)

    def __str__(self):
        return 'AugAssign({})'.format(self.lineprop)


class NodeBinOp(Node):
    def __init__(self, left: 'values.Value', right: 'values.Value', binop: 'BinOp', line=-1):
        super().__init__(line)

        #left = remove_ref(left)
        #right = remove_ref(right)

        self.left = left
        self.right = right
        self.binop = binop

        self.append_inputs(left)
        self.append_inputs(right)

    def __str__(self):
        return 'BinOp({},{})'.format(self.lineprop, self.binop)


class NodeUnaryOp(Node):
    def __init__(self, operand: 'values.Value', unaryop: 'UnaryOpType', line=-1):
        super().__init__(line)
        self.operand = operand
        self.unaryop = unaryop

        self.inputs.append(operand)

    def __str__(self):
        return 'UnaryOp({},{})'.format(self.lineprop, self.unaryop)

class NodeMultiaryOp(Node):
    def __init__(self, values_list: 'values.Value', multiaryop: 'MultiaryOpType', line=-1):
        super().__init__(line)
        self.values_list = values_list
        self.multiaryop = multiaryop

        self.extend_inputs(values_list)

    def __str__(self):
        return 'MultiaryOp({},{})'.format(self.lineprop, self.multiaryop)

class NodeCompare(Node):
    def __init__(self, left: 'values.Value', right: 'values.Value', compare: 'CompareType', line=-1):
        super().__init__(line)
        self.left = left
        self.right = right
        self.compare = compare

        self.append_inputs(left)
        self.append_inputs(right)

    def __str__(self):
        return 'Compare({},{})'.format(self.lineprop, self.compare)


class NodeGetItem(Node):
    def __init__(self, target: "values.Value", indexes, line=-1):
        super().__init__(line)
        self.target = target
        self.indexes = indexes

        self.append_inputs(target)
        self.extend_inputs(indexes)

    def __str__(self):
        return 'GetItem({})'.format(self.lineprop)


class NodeSlice(Node):
    def __init__(self, target: "values.Value", indices, slice_specs, line=-1):
        super().__init__(line)
        self.target = target
        self.indices = indices
        self.slice_specs = slice_specs

        self.append_inputs(target)
        self.extend_inputs(indices)

    def __str__(self):
        return 'Slice({})'.format(self.lineprop)


class NodeCall(Node):
    def __init__(self, func: 'Function', args : 'functions.FunctionArgInput', line=-1):
        super().__init__(line)
        args_ = args.get_value()
        attribute_args_ = args.get_value()

        self.func = func

        # args (it somtimes contains tuple(valueref))
        self.args = args_ # functions.FunctionArgValueInput

        # args for attributes (valuerefs are removed)
        self.attribute_args = make_attribute(attribute_args_)
        self.inputs.extend(self.args.inputs)

    def __str__(self):
        if self.func is not None and isinstance(self.func, values.FuncValue):
            return 'Call({}, {})'.format(self.lineprop, self.func.name)
        elif self.func is not None:
            return 'Call({}, {})'.format(self.lineprop, self.func.name)
        else:
            return 'Call({}, {})'.format(self.lineprop, 'Unknown')


class NodeReturn(Node):
    def __init__(self, value, line=-1):
        super().__init__(line)
        self.value = value
        self.append_inputs(value)

    def __str__(self):
        return 'Return({})'.format(self.lineprop)


class NodeIf(Node):
    def __init__(self, cond, input_values, true_graph, false_graph, line=-1):
        super().__init__(line)
        self.cond = cond
        self.input_values = input_values

        self.append_inputs(self.cond)
        self.extend_inputs(self.input_values)

        self.true_graph = true_graph
        self.false_graph = false_graph

        self.subgraphs.append(self.true_graph)
        self.subgraphs.append(self.false_graph)

    def __str__(self):
        return 'If({})'.format(self.lineprop)


class NodeFor(Node):
    def __init__(self, iter_value, input_values, body_graph, exit_cond, line=-1):
        super().__init__(line)
        self.iter_value = iter_value
        self.exit_cond = exit_cond
        self.input_values = input_values
        self.append_inputs(iter_value)
        self.extend_inputs(self.input_values)

        self.body_graph = body_graph
        self.subgraphs.append(self.body_graph)

    def __str__(self):
        return 'For({})'.format(self.lineprop)


class NodeForGenerator(Node):
    def __init__(self, counter_value, iter_value, line=-1):
        super().__init__(line)
        self.counter_value = counter_value
        self.iter_value = iter_value
        self.append_inputs(counter_value)
        self.append_inputs(iter_value)

    def __str__(self):
        return 'ForGen({})'.format(self.lineprop)


class NodeListcomp(Node):
    def __init__(self, iter_value, input_values, body_graph, line=-1):
        super().__init__(line)
        self.iter_value = iter_value
        self.input_values = input_values
        self.append_inputs(iter_value)
        self.extend_inputs(self.input_values)

        self.body_graph = body_graph
        self.subgraphs.append(self.body_graph)

    def __str__(self):
        return 'Listcomp({})'.format(self.lineprop)


class NodeGenerate(Node):
    def __init__(self, classtype, args, line=-1):
        super().__init__(line)
        if isinstance(args, list):
            self.extend_inputs(args)
            self.args = args
        else:
            args_ = args.get_value()
            self.args = args_
            self.extend_inputs(self.args.inputs)

        self.classtype = classtype

    def __str__(self):
        return 'Generate({},{})'.format(self.classtype, self.lineprop)


class NodeConvert(Node):
    def __init__(self, classtype, value, line=-1):
        super().__init__(line)
        self.classtype = classtype
        self.value = value
        self.append_inputs(self.value)

    def __str__(self):
        return 'Convert({},{})'.format(self.classtype, self.lineprop)

class NodeLen(Node):
    def __init__(self, iter_value, line=-1):
        super().__init__(line)
        self.iter_value = iter_value
        self.append_inputs(self.iter_value)

    def __str_(self):
        return 'Len({},{})'.format(self.classtype, self.lineprop)
