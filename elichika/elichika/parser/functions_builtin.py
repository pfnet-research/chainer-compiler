from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions

import chainer
import chainer.functions as F
import chainer.links as L

class ReluFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'relu'
        self.analyze_args(F.relu)

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'Value', args = [], line = -1):
        funcArgs = self.parse_args(args)
        node = nodes.NodeCall(self, [v.value for v in funcArgs], line)
        graph.add_node(node)
        value = functions.generateValueWithSameType(funcArgs[0].value)
        node.set_outputs([value])
        return value

class SoftmaxFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'softmax'
        self.analyze_args(F.softmax)

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'Value', args = [], line = -1):
        funcArgs = self.parse_args(args)
        node = nodes.NodeCall(self, [v.value for v in funcArgs], line)
        graph.add_node(node)
        value = functions.generateValueWithSameType(funcArgs[0].value)
        node.set_outputs([value])
        return value

class RangeFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'range'

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'Value', args = [], line = -1):
        node = nodes.NodeCall(self, [v.value for v in args], line)
        graph.add_node(node)
        value = values.Value()
        node.set_outputs([value])
        return value
