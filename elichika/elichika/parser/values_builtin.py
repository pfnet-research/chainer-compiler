from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser.graphs import Graph

import chainer.links

chainer_links = {}

class ChainerLinkDefinition:
    def __init__(self, estimate_shape = None):
        self.estimate_shape = estimate_shape

def estimate_linear_shape(inst : 'chainer.links.Linear', args):
    if isinstance(args[0].obj.get_value(), values.TensorValue) and len(args[0].obj.get_value().shape) >= 2:
        return (args[0].obj.get_value().shape[0], inst.out_size)

def estimate_convolution2D_shape(inst : 'chainer.links.Convolution2D', args):
    return functions.generate_tensor_value_with_undefined_shape_size(args[0].obj.get_value()).shape


chainer_links[chainer.links.Linear] = ChainerLinkDefinition(estimate_linear_shape)
chainer_links[chainer.links.Convolution2D] = ChainerLinkDefinition(estimate_convolution2D_shape)

def is_builtin_chainer_link(value) -> 'bool':
    return type(value) in chainer_links.keys()

class ChainerLinkFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = '__call__'
        self.owner = owner

    def vcall(self, module : 'values.Field', graph : 'Graph', inst : 'Object', args = [], line = -1):
        node = nodes.NodeCall(self, [v.obj.get_value() for v in args], line)
        graph.add_node(node)
        value = values.TensorValue()

        estimate_shape = chainer_links[type(self.owner.inst)].estimate_shape
        if estimate_shape is not None:
            value.shape = estimate_shape(self.owner.inst, args)

        node.set_outputs([value])
        return values.Object(value)

class ChainerLinkInstance(values.Instance):
    def __init__(self, module : 'Field', inst):
        super().__init__(module, inst, None)
        self.callable = True

    def apply_to_object(self, obj : 'values.Object'):
        self.func = values.Object(values.FuncValue(ChainerLinkFunction(self), obj))
        obj.get_field().get_attribute('forward').revise(self.func)
