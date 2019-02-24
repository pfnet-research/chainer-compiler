from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import graphs

import chainer
import chainer.functions as F
import chainer.links as L

class ReluFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'relu'
        self.analyze_args(F.relu)

    def vcall(self, module : 'Field', graph : 'Graph', inst : 'values.Object', args = [], line = -1):
        funcArgs = self.parse_args(args)
        vargs = self.get_values(funcArgs)

        node = nodes.NodeCall(self, vargs, line)
        graph.add_node(node)
        value = functions.generate_value_with_same_type(vargs[0])
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.Object(value)

class SoftmaxFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'softmax'
        self.analyze_args(F.softmax)

    def vcall(self, module : 'Field', graph : 'Graph', inst : 'values.Object', args = [], line = -1):
        funcArgs = self.parse_args(args)
        vargs = self.get_values(funcArgs)

        node = nodes.NodeCall(self, vargs, line)
        graph.add_node(node)
        value = functions.generate_value_with_same_type(vargs[0])
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.Object(value)

class SoftmaxCrossEntropyFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'softmax_cross_entropy'
        self.analyze_args(F.softmax_cross_entropy)

    def vcall(self, module : 'Field', graph : 'Graph', inst : 'values.Object', args = [], line = -1):
        funcArgs = self.parse_args(args)
        vargs = self.get_values(funcArgs)

        node = nodes.NodeCall(self, vargs, line)
        graph.add_node(node)
        value = functions.generate_value_with_same_type(vargs[0])
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.Object(value)

class RangeFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'range'

    def vcall(self, module : 'Field', graph : 'Graph', inst : 'values.Object', args = [], line = -1):
        node = nodes.NodeGenerate('range', [v.obj.get_value() for v in args], line)
        graph.add_node(node)
        value = values.RangeValue()
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.Object(value)

class AppendFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = 'append'
        self.owner = owner

    def vcall(self, module : 'Field', graph : 'Graph', inst : 'values.Object', args = [], line = -1):
        assert(len(args) == 1)

        node = nodes.NodeCall(self, [inst.get_value()] + [v.obj.get_value() for v in args], line)

        old_v = inst.get_value()
        new_v = functions.generate_value_with_same_type(old_v)
        inst.revise(new_v)

        new_v.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([new_v])

        graph.add_node(node)
        return values.NoneValue()
