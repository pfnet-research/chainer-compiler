import chainer
import chainer.functions as F
import chainer.links as L
import inspect
from enum import Enum

from elichika.parser import core
from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import utils

class BinOpType(Enum):
    Add = 0,
    Sub = 1,
    Mul = 2,
    Unknown = 255,

class UnaryOpType(Enum):
    UAdd = 0,
    USub = 1,
    Not = 2,
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
            output.generator = self

class NodeCopy(Node):
    def __init__(self, value : 'values.Value', line = -1):
        super().__init__(line)
        self.value = value
        self.append_inputs(value)

    def __str__(self):
        return 'Copy({})'.format(self.lineprop)

class NodeNonVolatileAssign(Node):
    def __init__(self, target_value : 'values.Value', value : 'values.Value', line = -1):
        super().__init__(line)
        self.target_value = target_value
        self.value = value
        self.append_inputs(target_value)
        self.append_inputs(value)

    def __str__(self):
        return 'NodeNonVolatileAssign({})'.format(self.lineprop)

class NodeAssign(Node):
    def __init__(self, attr : 'values.Attribute', obj : 'values.Object', line = -1):
        assert(isinstance(obj,values.Object))
        
        super().__init__(line)
        self.targets = []
        self.objects = []

        self.targets.append(attr)
        self.objects.append(obj)

    def __str__(self):
        return 'Assign({})'.format(self.lineprop)

class NodeAugAssign(Node):
    def __init__(self, target : 'values.Value', value : 'values.Value', binop : 'BinOp', line = -1):
        super().__init__(line)
        self.target = target
        self.value = value
        self.binop = binop

        self.append_inputs(target)
        self.append_inputs(value)

    def __str__(self):
        return 'AugAssign({})'.format(self.lineprop)

class NodeBinOp(Node):
    def __init__(self, left : 'values.Value', right : 'values.Value', binop : 'BinOp', line = -1):
        super().__init__(line)
        self.left = left
        self.right = right
        self.binop = binop

        self.append_inputs(left)
        self.append_inputs(right)

    def __str__(self):
        return 'BinOp({},{})'.format(self.lineprop, self.binop)

class NodeUnaryOp(Node):
    def __init__(self, operand : 'values.Value', unaryop : 'UnaryOpType', line = -1):
        super().__init__(line)
        self.operand = operand
        self.unaryop = unaryop

        self.inputs.append(operand)
    def __str__(self):
        return 'UnaryOp({},{})'.format(self.lineprop, self.unaryop)

class NodeCompare(Node):
    def __init__(self, left : 'values.Value', right : 'values.Value', compare : 'CompareType', line = -1):
        super().__init__(line)
        self.left = left
        self.right = right
        self.compare = compare

        self.append_inputs(left)
        self.append_inputs(right)

    def __str__(self):
        return 'Compare({},{})'.format(self.lineprop, self.compare)

class NodeGetItem(Node):
    def __init__(self, target : "values.Value", indexes, line = -1):
        super().__init__(line)
        self.target = target
        self.indexes = indexes

        self.append_inputs(target)
        self.extend_inputs(indexes)

    def __str__(self):
        return 'GetItem({})'.format(self.lineprop)

class NodeSlice(Node):
    def __init__(self, target : "values.Value", indices, slice_specs, line = -1):
        super().__init__(line)
        self.target = target
        self.indices = indices
        self.slice_specs = slice_specs

        self.append_inputs(target)
        self.extend_inputs(indices)

    def __str__(self):
        return 'Slice({})'.format(self.lineprop)

class NodeCall(Node):
    def __init__(self, func : 'Function', args, line = -1):
        super().__init__(line)
        self.func = func
        self.args = args
        self.inputs.extend(self.args)

    def __str__(self):
        if self.func is not None and isinstance(self.func, values.FuncValue):
            return 'Call({}, {})'.format(self.lineprop, self.func.name)
        elif self.func is not None:
            return 'Call({}, {})'.format(self.lineprop, self.func.name)
        else:
            return 'Call({}, {})'.format(self.lineprop, 'Unknown')

class NodeReturn(Node):
    def __init__(self, value, line = -1):
        super().__init__(line)
        self.value = value
        self.append_inputs(value)

    def __str__(self):
        return 'Return({})'.format(self.lineprop)

class NodeIf(Node):
    def __init__(self, cond, input_values, true_graph, false_graph, line = -1):
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
    def __init__(self, iter_value, input_values, body_graph, line = -1):
        super().__init__(line)
        self.iter_value = iter_value
        self.input_values = input_values
        self.append_inputs(iter_value)
        self.extend_inputs(self.input_values)

        self.body_graph = body_graph
        self.subgraphs.append(self.body_graph)

    def __str__(self):
        return 'For({})'.format(self.lineprop)

class NodeForGenerator(Node):
    def __init__(self, counter_value, iter_value, line = -1):
        super().__init__(line)
        self.counter_value = counter_value
        self.iter_value = iter_value
        self.append_inputs(counter_value)
        self.append_inputs(iter_value)

    def __str__(self):
        return 'ForGen({})'.format(self.lineprop)


class NodeListcomp(Node):
    def __init__(self, iter_value, input_values, body_graph, line = -1):
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
    def __init__(self, classtype, args, line = -1):
        super().__init__(line)
        self.classtype = classtype
        self.args = args
        self.extend_inputs(self.args)

    def __str__(self):
        return 'Generate({},{})'.format(self.classtype, self.lineprop)
