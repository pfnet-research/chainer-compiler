from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import graphs

import chainer.links

chainer_links = {}


class ChainerLinkDefinition:
    def __init__(self, args=None, get_ret=None, estimate_shape=None):
        self.estimate_shape = estimate_shape
        self.args = args
        self.get_ret = get_ret


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

def estimate_EmbedID_shape(inst: 'chainer.links.BatchNormalization', args: 'functions.FunctionArgInput'):
    if isinstance(args.get_value().get_value('x'), values.TensorValue):
        return args.get_value().get_value('x').shape
    return ()

def estimate_NStepLSTM_shape(inst: 'chainer.links.EmbedID', args: 'functions.FunctionArgInput'):
    inst.out_size
    if isinstance(args.get_value().get_value('x'), values.TensorValue):
        return args.get_value().get_value('x').shape
    return ()

def estimate_NStepBiLSTM_shape(inst: 'chainer.links.NStepBiLSTM', args: 'functions.FunctionArgInput'):
    inst.out_size
    if isinstance(args.get_value().get_value('x'), values.TensorValue):
        return args.get_value().get_value('x').shape
    return ()

def return_NStepLSTM():
    list_tensor = values.ListValue()
    list_tensor.vtype = values.TensorValue
    return [values.TensorValue(), values.TensorValue(), list_tensor]

def return_NStepBiLSTM():
    list_tensor = values.ListValue()
    list_tensor.vtype = values.TensorValue
    return [values.TensorValue(), values.TensorValue(), list_tensor]

chainer_links[chainer.links.Linear] = ChainerLinkDefinition(
    args=[('self', values.NoneValue()), ('x', values.NoneValue()), ('n_batch_axes',values.NumberValue(1))],
    estimate_shape=estimate_linear_shape)
chainer_links[chainer.links.Convolution2D] = ChainerLinkDefinition(
    estimate_shape=estimate_convolution2D_shape)
chainer_links[chainer.links.BatchNormalization] = ChainerLinkDefinition(
    estimate_shape=estimate_batch_norm_shape)
chainer_links[chainer.links.NStepLSTM] = ChainerLinkDefinition(
    args=[('self', values.NoneValue()), ('hx', values.NoneValue()), ('cx', values.NoneValue()),
          ('xs', values.NoneValue())],
    estimate_shape=estimate_NStepLSTM_shape,
    get_ret=return_NStepLSTM)
chainer_links[chainer.links.NStepBiLSTM] = ChainerLinkDefinition(
    args=[('self', values.NoneValue()), ('hx', values.NoneValue()), ('cx', values.NoneValue()),
          ('xs', values.NoneValue())],
    estimate_shape=estimate_NStepBiLSTM_shape,
    get_ret=return_NStepBiLSTM)
chainer_links[chainer.links.EmbedID] = ChainerLinkDefinition(
    estimate_shape=estimate_EmbedID_shape)


def is_builtin_chainer_link(value) -> 'bool':
    return type(value) in chainer_links.keys()


class ChainerLinkFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = '__call__'
        self.owner = owner

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):

        chainer_link = chainer_links[type(self.owner.inst)]

        if len(self.args.args_list) == 0:
            if chainer_link.args is None:
                self.args.add_arg('self', None)
                self.args.add_arg('x', None)
            else:
                for arg in chainer_link.args:
                    self.args.add_arg(arg[0], arg[1])

        vargs = self.args.merge_inputs(inst, args)

        node = nodes.NodeCall(self, vargs, line)
        graph.add_node(node)

        if chainer_link.get_ret is not None:
            ret = chainer_link.get_ret()
            node.set_outputs(ret)
            return values.ValueRef(values.TupleValue([values.ValueRef(v) for v in ret]))
        else:
            value = values.TensorValue()

            estimate_shape = chainer_links[type(
                self.owner.inst)].estimate_shape
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

        obj.attributes.set_predefined_obj('__call__', callable_func)
        obj.attributes.set_predefined_obj('forward', callable_func)


class ChainerChainListChildrenFunction(functions.FunctionBase):
    def __init__(self, owner):
        super().__init__()
        self.name = 'children'
        self.owner = owner

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalOption' = None, line=-1):
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
        obj.attributes.set_predefined_obj('children', children)

        forward_func = obj.try_get_and_store_obj('forward', None)
        if forward_func is not None:
            obj.attributes.set_predefined_obj('__call__', forward_func)
            obj.attributes.set_predefined_obj('forward', forward_func)


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

        forward_func = obj.try_get_and_store_obj('forward', None)
        if forward_func is not None:
            obj.attributes.set_predefined_obj('__call__', forward_func)
            obj.attributes.set_predefined_obj('forward', forward_func)
