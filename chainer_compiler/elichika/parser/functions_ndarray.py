from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import graphs
from chainer_compiler.elichika.parser import utils

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class NDArrayInt32(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'int32'
        self.dtype = np.int32
    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        assert(inst is None)
        return values.ValueRef(values.NoneValue)

class NDArrayFloat32(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'float32'
        self.dtype = np.float32

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        assert(inst is None)
        return values.ValueRef(values.NoneValue)

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

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        dtype_value = vargs[1]
        if isinstance(dtype_value, values.StrValue):
            if not dtype_value.has_constant_value():
                utils.print_error('Failed to get dtype str ', line)
                return None

            dtype = utils.str_2_dtype(dtype_value.get_constant_value())

        elif dtype_value is not None and not isinstance(dtype_value, values.NoneValue):
            # TODO : make better
            dtype = np.array(1, dtype=dtype_value.func.dtype).dtype
        elif isinstance(vargs[0], values.TensorValue):
            dtype = vargs[0].dtype
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

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        dtype_value = vargs[1]
        if isinstance(dtype_value, values.StrValue):
            if not dtype_value.has_constant_value():
                utils.print_error('Failed to get dtype str ', line)
                return None

            dtype = utils.str_2_dtype(dtype_value.get_constant_value())

        elif dtype_value is not None and not isinstance(dtype_value, values.NoneValue):
            # TODO : make better
            dtype = np.array(1, dtype=dtype_value.func.dtype).dtype
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

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        dtype_value = vargs[2]
        if isinstance(dtype_value, values.StrValue):
            if not dtype_value.has_constant_value():
                utils.print_error('Failed to get dtype str ', line)
                return None

            if dtype_value.get_constant_value() == 'q':
                dtype = np.int64
            elif dtype_value.get_constant_value() == 'i':
                dtype = np.int32
            elif dtype_value.get_constant_value() == 'g':
                dtype = np.float64
            elif dtype_value.get_constant_value() == 'f':
                dtype = np.float32
            else:
                assert(False)
        elif dtype_value is not None and not isinstance(dtype_value, values.NoneValue):
            # TODO : make better
            dtype = np.array(1, dtype=dtype_value.func.dtype).dtype
        else:
            dtype = np.array(vargs[1].internal_value).dtype

        node = nodes.NodeGenerate('full', funcArgs, line)
        graph.add_node(node)
        value = values.TensorValue()
        value.dtype = dtype
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)

class NDArrayCeilFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'ceil'
        self.args.add_arg('x', values.NoneValue())

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        node = nodes.NodeCall(self, funcArgs, line)
        graph.add_node(node)
        value = functions.generate_value_with_same_type(vargs[0])
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)

class NDArrayCumsumFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'cumsum'
        self.args.add_arg('a', values.NoneValue())
        self.args.add_arg('axis', values.NoneValue())
        self.args.add_arg('dtype', values.NoneValue())
        self.args.add_arg('out', values.NoneValue())

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        assert(inst is None)

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        node = nodes.NodeCall(self, funcArgs, line)
        graph.add_node(node)
        value = functions.generate_value_with_same_type(vargs[0])
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return values.ValueRef(value)

class NDArrayShapeFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'shape'
        self.is_property = True

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
        args = functions.FunctionArgInput()
        args.inputs.append(inst)
        args.keywords['self'] = inst

        node = nodes.NodeCall(self, args, line)

        value = values.TupleValue()
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])

        graph.add_node(node)
        return values.ValueRef(value)

class NDArraySizeFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'size'
        self.is_property = True

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
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
