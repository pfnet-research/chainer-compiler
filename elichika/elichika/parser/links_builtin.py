from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser.graphs import Graph

import chainer.links

chainer_links = {}


class ChainerLinkDefinition:
    def __init__(self, estimate_shape=None):
        self.estimate_shape = estimate_shape


def estimate_linear_shape(inst: 'chainer.links.Linear', args: 'functions.FunctionArgInput'):
    if isinstance(args.get_value().get_value('x'), values.TensorValue) and len(args.get_value().get_value('x').shape) >= 2:
        return (args.get_value().get_value('x').shape[0], inst.out_size)
    return ()

def estimate_convolution2D_shape(inst: 'chainer.links.Convolution2D', args: 'functions.FunctionArgInput'):
    # TODO make correct
    return functions.generate_tensor_value_with_undefined_shape_size(args.get_value().get_value('x')).shape

def estimate_batch_norm_shape(inst: 'chainer.links.BatchNormalization', args: 'functions.FunctionArgInput'):
    if isinstance(args.get_value().get_value('x'), values.TensorValue):
        return args.get_value().get_value('x').shape
    return ()
    
chainer_links[chainer.links.Linear] = ChainerLinkDefinition(
    estimate_linear_shape)
chainer_links[chainer.links.Convolution2D] = ChainerLinkDefinition(
    estimate_convolution2D_shape)
chainer_links[chainer.links.BatchNormalization] = ChainerLinkDefinition(
    estimate_batch_norm_shape)


def is_builtin_chainer_link(value) -> 'bool':
    return type(value) in chainer_links.keys()


class ChainerLinkFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = '__call__'
        self.owner = owner
        self.args.add_arg('self', None)
        self.args.add_arg('x', None)

    def vcall(self, module: 'values.Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        vargs = self.args.merge_inputs(inst, args)

        node = nodes.NodeCall(self, vargs, line)
        graph.add_node(node)
        value = values.TensorValue()

        estimate_shape = chainer_links[type(self.owner.inst)].estimate_shape
        if estimate_shape is not None:
            value.shape = estimate_shape(self.owner.inst, vargs)

        node.set_outputs([value])
        return values.ValueRef(value)


class ChainerLinkInstance(values.Instance):
    def __init__(self, module: 'Field', inst):
        super().__init__(module, inst, None)

    def apply_to_object(self, obj: 'values.ValueRef'):
        callable_func = values.ValueRef(
            values.FuncValue(ChainerLinkFunction(self), obj))

        obj.get_field().get_attribute('__call__').revise(callable_func)
        obj.get_field().get_attribute('forward').revise(callable_func)

class ChainerChainListChildrenFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = 'children'
        self.owner = owner

    def vcall(self, module: 'Field', graph: 'Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput', line=-1):
        args = functions.FunctionArgInput()
        args.inputs.append(inst)
        args.keywords['self'] = inst

        value = values.ListValue(self.owner.children)
        return values.ValueRef(value)


class ChainerChainListInstance(values.UserDefinedInstance):
    def __init__(self, module: 'Field', inst):
        super().__init__(module, inst, None)
        self.is_chainer_link = True
        self.children = []

        for child in inst.children():
            child_ = values.parse_instance(module, '', child, inst)
            self.children.append(child_)

    def apply_to_object(self, obj: 'values.ValueRef'):
        super().apply_to_object(obj)
        children = values.ValueRef(
            values.FuncValue(ChainerChainListChildrenFunction(self), obj))
        obj.get_field().get_attribute('children').revise(children)

        forward_func = obj.try_get_and_store_obj('forward')
        if forward_func is not None:
            obj.get_field().get_attribute('__call__').revise(forward_func)
            obj.get_field().get_attribute('forward').revise(forward_func)

class ChainerChainInstance(values.UserDefinedInstance):
    def __init__(self, module: 'Field', inst):
        super().__init__(module, inst, None)
        self.is_chainer_link = True
        self.children = []

        for child in inst.children():
            child_ = values.parse_instance(module, '', child, inst)
            self.children.append(child_)

    def apply_to_object(self, obj: 'values.ValueRef'):
        super().apply_to_object(obj)
        children = values.ValueRef(
            values.FuncValue(ChainerChainListChildrenFunction(self), obj))
        obj.get_field().get_attribute('children').revise(children)

        forward_func = obj.try_get_and_store_obj('forward')
        if forward_func is not None:
            obj.get_field().get_attribute('__call__').revise(forward_func)
            obj.get_field().get_attribute('forward').revise(forward_func)
