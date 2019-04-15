from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import graphs
from elichika.parser import utils

import chainer
import chainer.functions as F
import chainer.links as L


class ChainerFunction(functions.FunctionBase):
    def __init__(self, func):
        super().__init__()
        self.name = str(func)
        self.args.analyze_args(func)
        self.base_func = func

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        funcArgs = self.args.merge_inputs(inst, args)

        node = nodes.NodeCall(self, funcArgs, line)
        graph.add_node(node)
        #value = functions.generate_value_with_same_type(vargs[0])
        value = values.TensorValue()
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


class ListFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'list'
        self.args.add_arg(functions.FunctionArg(
            'value', values.ValueRef(values.NoneValue())))

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
        self.args.add_arg(functions.FunctionArg('self'))
        self.args.add_arg(functions.FunctionArg('elmnt'))

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


class NDArrayFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'array'

        fa = functions.FunctionArg()
        fa.name = 'object'
        fa.obj = values.ValueRef(values.NoneValue())
        self.args.add_arg(fa)

        fa = functions.FunctionArg()
        fa.name = 'dtype'
        fa.obj = values.ValueRef(values.NoneValue())
        self.args.add_arg(fa)

        fa = functions.FunctionArg()
        fa.name = 'copy'
        fa.obj = values.ValueRef(values.BoolValue(True))
        self.args.add_arg(fa)

        fa = functions.FunctionArg()
        fa.name = 'order'
        fa.obj = values.ValueRef(values.StrValue('K'))
        self.args.add_arg(fa)

        fa = functions.FunctionArg()
        fa.name = 'subok'
        fa.obj = values.ValueRef(values.BoolValue(False))
        self.args.add_arg(fa)

        fa = functions.FunctionArg()
        fa.name = 'ndmin'
        fa.obj = values.ValueRef(values.NumberValue(0))
        self.args.add_arg(fa)

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        dtype_value = vargs[1]
        if dtype_value is not None and not isinstance(dtype_value, values.NoneValue):
            # TODO : make better
            dtype = utils.int_2_numpy_type(dtype_value.internal_value)
        else:
            dtype = None

        node = nodes.NodeGenerate('array', funcArgs, line)
        node.fargs = funcArgs
        graph.add_node(node)
        value = values.TensorValue()
        value.dtype = dtype
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)


class NDArrayShapeFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = 'shape'
        self.owner = owner
        self.is_property = True

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        args = functions.FunctionArgInput()
        args.inputs.append(inst)
        args.keywords['self'] = inst

        node = nodes.NodeCall(self, args, line)

        value = values.ListValue()
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])

        # TODO should make tuple
        graph.add_node(node)
        return values.ValueRef(value)
