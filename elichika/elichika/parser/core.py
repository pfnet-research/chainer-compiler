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

    model_inst = values.parse_instance(default_module, '', model)
    forward_func = model_inst.try_get_and_store_value('forward')

    # convert args
    value_args = []
    function_args = []
    for arg in args:
        varg = values.parse_instance(default_module, '', arg)
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

    for v in value_args:
        graph.add_input_value(v)

    for v in ret_:
        graph.add_output_value(v)

    return (value_args, ret_, graph)
