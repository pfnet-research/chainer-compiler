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
        value = functions.generate_value_with_same_type(funcArgs[0].value)
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
        value = functions.generate_value_with_same_type(funcArgs[0].value)
        node.set_outputs([value])
        return value

class RangeFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'range'

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'Value', args = [], line = -1):
        node = nodes.NodeGenerate('range', [v.value for v in args], line)
        graph.add_node(node)
        value = values.RangeValue()
        node.set_outputs([value])
        return value

class AppendFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = 'append'
        self.owner = owner

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'values.Value', args = [], line = -1):
        assert(len(args) == 1)

        node = nodes.NodeCall(self, [v.value for v in args], line)
        inst.modify(node, None)
        graph.add_node(node)
        return values.NoneValue()
