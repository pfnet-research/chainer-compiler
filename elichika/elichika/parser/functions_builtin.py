from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import graphs
from elichika.parser import utils

import chainer
import chainer.functions as F
import chainer.links as L

def create_return_value_in_chainer_function():
    return values.TensorValue()

class ChainerFunction(functions.FunctionBase):
    def __init__(self, func, ret_value_func = create_return_value_in_chainer_function):
        super().__init__()
        self.name = str(func)
        self.args.analyze_args(func)
        self.base_func = func
        self.ret_value_func = ret_value_func

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        funcArgs = self.args.merge_inputs(inst, args)

        node = nodes.NodeCall(self, funcArgs, line)
        graph.add_node(node)
        #value = functions.generate_value_with_same_type(vargs[0])
        value = self.ret_value_func()
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)


class RangeFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'range'

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        node = nodes.NodeGenerate(
            'range', [v.get_value() for v in args.inputs], line)
        graph.add_node(node)
        value = values.RangeValue()
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)


class LenFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'len'

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        node = nodes.NodeLen(
            args.inputs[0].get_value(),  # TODO: Check this.
            line
        )
        graph.add_node(node)
        value = values.NumberValue(None)
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)


class ListFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'list'
        self.args.add_arg('value', values.ValueRef(values.NoneValue()))

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst, args)
        vargs = funcArgs.get_value().inputs
        value = values.ListValue()

        if isinstance(vargs[0], values.NoneValue):
            node = nodes.NodeGenerate('List', [], line)
            graph.add_node(node)
        else:
            node = nodes.NodeConvert('List', vargs[0], line)
            graph.add_node(node)

        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)


class AppendFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = 'append'
        self.owner = owner
        self.args.add_arg('self', None)
        self.args.add_arg('elmnt', None)

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        funcArgs = self.args.merge_inputs(inst, args)

        node = nodes.NodeCall(self, funcArgs, line)

        old_v = inst.get_value()
        new_v = functions.generate_value_with_same_type(old_v)
        inst.revise(new_v)

        new_v.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([new_v])

        graph.add_node(node)
        return values.NoneValue()
