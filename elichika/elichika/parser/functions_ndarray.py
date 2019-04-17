from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import graphs
from elichika.parser import utils

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class NDArrayFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'array'

        self.args.add_arg('object', values.NoneValue())
        self.args.add_arg('dtype', values.NoneValue())
        self.args.add_arg('copy', values.BoolValue(True))
        self.args.add_arg('order', values.StrValue('K'))
        self.args.add_arg('subok', values.BoolValue(False))
        self.args.add_arg('ndmin', values.NumberValue(0))

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
        graph.add_node(node)
        value = values.TensorValue()
        value.dtype = dtype
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)

class NDArrayZerosFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'zeros'
        self.args.add_arg('shape', values.NoneValue())
        self.args.add_arg('dtype', values.NoneValue())
        self.args.add_arg('order', values.StrValue('C'))

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        dtype_value = vargs[1]
        if dtype_value is not None and not isinstance(dtype_value, values.NoneValue):
            # TODO : make better
            dtype = utils.int_2_numpy_type(dtype_value.internal_value)
        else:
            dtype = np.array(vargs[1].internal_value).dtype

        node = nodes.NodeGenerate('zeros', funcArgs, line)
        graph.add_node(node)
        value = values.TensorValue()
        value.dtype = dtype
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)

class NDArrayFullFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'full'
        self.args.add_arg('shape', values.NoneValue())
        self.args.add_arg('fill_value', values.NoneValue())
        self.args.add_arg('dtype', values.NoneValue())
        self.args.add_arg('order', values.StrValue('C'))

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        dtype_value = vargs[2]
        if dtype_value is not None and not isinstance(dtype_value, values.NoneValue):
            # TODO : make better
            dtype = utils.int_2_numpy_type(dtype_value.internal_value)
        else:
            dtype = np.array(vargs[1].internal_value).dtype

        node = nodes.NodeGenerate('full', funcArgs, line)
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

class NDArraySizeFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = 'size'
        self.owner = owner
        self.is_property = True

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        args = functions.FunctionArgInput()
        args.inputs.append(inst)
        args.keywords['self'] = inst

        node = nodes.NodeCall(self, args, line)

        value = values.NumberValue(None)
        value.dtype = np.array(0).dtype
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])

        graph.add_node(node)
        return values.ValueRef(value)
