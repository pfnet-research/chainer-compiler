import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import weakref
from elichika.parser import config
from elichika.parser import nodes
from elichika.parser import vevaluator
from elichika.parser import values
from elichika.parser import values_builtin
from elichika.parser import functions
from elichika.parser import functions_builtin
from elichika.parser import utils
from elichika.parser.graphs import Graph
import numpy as np


def parse_instance(default_module, name, instance):

    if values_builtin.is_builtin_chainer_link(instance):
        return values_builtin.ChainerLinkInstance(default_module, instance)

    # need to check whether is value bool before check whether is value int
    if isinstance(instance, bool):
        return values.BoolValue(instance)

    if isinstance(instance, int) or isinstance(instance, float):
        return values.NumberValue(instance)

    if isinstance(instance, str):
        return values.StrValue(instance)

    if isinstance(instance, list):
        ret = values.ListValue()
        ind = 0
        for e in instance:
            element_value = parse_instance(default_module, '', e)
            ret.get_field().get_attribute(str(ind)).revise(element_value)
            ind += 1
        return ret

    if isinstance(instance, tuple) and 'Undefined' in instance:
        shape = list(instance)
        shape = -1 if shape == 'Undefined' else shape
        tensorValue = values.TensorValue()
        tensorValue.shape = tuple(shape)
        return tensorValue

    if isinstance(instance, np.ndarray):
        tensorValue = values.TensorValue()
        tensorValue.value = instance
        tensorValue.shape = instance.shape
        return tensorValue

    if instance is None:
        return values.NoneValue()

    if not isinstance(instance, chainer.Link):
        if config.show_warnings:
            print('Warning unsupported format is found : {}, {}'.format(name, instance))
        return values.NoneValue()

    model_inst = values.UserDefinedInstance(default_module, instance)

    for attr_k, attr_v in instance.__dict__.items():
        attr_inst = parse_instance(default_module, attr_k, attr_v)
        model_inst.get_field().get_attribute(attr_k).revise(attr_inst)

    return model_inst

def convert_model(model : 'chainer.Chain', args = []):
    # reset values
    values.reset_field_and_attributes()
    utils.reset_guid()
    
    # generate default module
    default_module = values.Field(None, None)
    f_dict = values.DictValue()
    f_relu = values.FuncValue(functions_builtin.ReluFunction(), None)
    f_dict.get_field().get_attribute('relu').revise(f_relu)
    f_softmax = values.FuncValue(functions_builtin.SoftmaxFunction(), None)
    f_dict.get_field().get_attribute('softmax').revise(f_softmax)
    default_module.get_attribute('F').revise(f_dict)
    m_range = values.FuncValue(functions_builtin.RangeFunction(), None)
    default_module.get_attribute('range').revise(m_range)

    model_inst = parse_instance(default_module, '', model)
    forward_func = model_inst.try_get_func('forward')

    # convert args
    value_args = []
    function_args = []
    for arg in args:
        varg = parse_instance(default_module, '', arg)
        farg = functions.FunctionArg()
        farg.value = varg
        value_args.append(varg)
        function_args.append(farg)    

    graph = Graph()

    ret = forward_func.func.vcall(default_module, graph, forward_func.value, function_args)

    ret_ = []
    if isinstance(ret, values.TupleValue):
        ret_.extend([v.get_value() for v in ret.values])
    elif not isinstance(ret, list):
        ret_ = [ret]
    else:
        ret_ = ret

    return (value_args, ret_, graph)
