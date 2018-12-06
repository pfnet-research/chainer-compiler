import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import ast, gast
from enum import Enum

from elichika.parser import core
from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import utils

class BinOpType(Enum):
    Add = 0,
    Sub = 1,
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
        self.onnx_name : str = ""

        if isinstance(line, int):
            self.lineprop.lineno = line
        else:
            self.lineprop = line

        return

    def set_outputs(self, outputs):
        self.outputs = outputs

        for output in self.outputs:
            output.generator = self

class NodeAssign(Node):
    def __init__(self, attr : 'values.Attribute', value : 'values.Value', line = -1):
        super().__init__(line)
        self.targets = []
        self.values = []

        self.targets.append(attr)
        self.values.append(value)

    def __str__(self):
        return 'Assign({})'.format(self.lineprop)

class NodeAugAssign(Node):
    def __init__(self, target : 'values.Value', value : 'values.Value', line = -1):
        super().__init__(line)
        self.target = target 
        self.value = value

        self.inputs.append(target)
        self.inputs.append(value)

    def __str__(self):
        return 'AugAssign({})'.format(self.lineprop)

class NodeBinOp(Node):
    def __init__(self, left : 'values.Value', right : 'values.Value', binop : 'BinOp', line = -1):
        super().__init__(line)
        self.left = left 
        self.right = right
        self.binop = binop

        self.inputs.append(left)
        self.inputs.append(right)

    def __str__(self):
        return 'BinOp({},{})'.format(self.lineprop, self.binop)

class NodeCompare(Node):
    def __init__(self, left : 'values.Value', right : 'values.Value', compare : 'CompareType', line = -1):
        super().__init__(line)
        self.left = left 
        self.right = right
        self.compare = compare

        self.inputs.append(left)
        self.inputs.append(right)

    def __str__(self):
        return 'Compare({},{})'.format(self.lineprop, self.compare)

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
        self.inputs.append(value)

    def __str__(self):
        return 'Return({})'.format(self.lineprop)

class NodeIf(Node):
    def __init__(self, cond, input_values, true_graph, false_graph, line = -1):
        super().__init__(line)
        self.cond = cond
        self.input_values = input_values
        
        self.inputs.append(self.cond)
        self.inputs.extend(self.input_values)
        
        self.true_graph = true_graph
        self.false_graph = false_graph

        self.subgraphs.append(self.true_graph)
        self.subgraphs.append(self.false_graph)
        
    def __str__(self):
        return 'If({})'.format(self.lineprop)

class NodeIfOutput(Node):
    def __init__(self, input_value, line = -1):
        super().__init__(line)
        self.input_value = input_value
        self.inputs.append(input_value)

    def __str__(self):
        return 'IfOutput({})'.format(self.lineprop)

class NodeFor(Node):
    def __init__(self, iter_value, input_values, body_graph, line = -1):
        super().__init__(line)
        self.iter_value = iter_value
        self.input_values = input_values
        self.inputs.append(iter_value)
        self.inputs.extend(self.input_values)
        
        self.body_graph = body_graph
        self.subgraphs.append(self.body_graph)
        
    def __str__(self):
        return 'For({})'.format(self.lineprop)

class NodeForOutput(Node):
    def __init__(self, input_value, line = -1):
        super().__init__(line)
        self.input_value = input_value
        self.inputs.append(input_value)

    def __str__(self):
        return 'ForOutput({})'.format(self.lineprop)
