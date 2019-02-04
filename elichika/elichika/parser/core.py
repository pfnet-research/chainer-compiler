import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import weakref
import sys
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

def get_module_name(target_module, parent_module):
    members = inspect.getmembers(parent_module)

    for member in members:
        if member[1] is target_module:
            return member[0]

    return ''

def convert_model(model : 'chainer.Chain', args = []):
    # reset values
    values.reset_field_and_attributes()
    utils.reset_guid()

    # generate default module
    default_module = values.Module(sys.modules[model.__module__])

    # chainer.functions
    chainer_functions_module_name = get_module_name(F, default_module.internal_module)

    if chainer_functions_module_name != '':
        f_dict = values.Object(values.ModuleValue())
        f_relu = values.FuncValue(functions_builtin.ReluFunction(), None)
        f_dict.get_field().get_attribute('relu').revise(values.Object(f_relu))
        f_softmax = values.FuncValue(functions_builtin.SoftmaxFunction(), None)
        f_dict.get_field().get_attribute('softmax').revise(values.Object(f_softmax))
        default_module.set_default_value(chainer_functions_module_name, f_dict)

    m_range = values.FuncValue(functions_builtin.RangeFunction(), None)
    default_module.set_default_value('range', values.Object(m_range))

    model_inst = values.parse_instance(default_module, '', model)
    forward_func = model_inst.try_get_and_store_obj('forward')

    # convert args
    value_args = []
    function_args = []
    ind = 0
    for arg in args:
        varg = values.parse_instance(default_module, '', arg)
        varg.name = 'in_' + str(ind)
        varg.get_value().name = 'in_' + str(ind)
        farg = functions.FunctionArg()
        farg.obj = varg
        value_args.append(varg.get_value())
        function_args.append(farg)
        ind += 1

    graph = Graph()
    forward_func_value = forward_func.get_value()
    ret = forward_func_value.func.vcall(default_module, graph, forward_func_value.obj, function_args)
    assert(ret is None or isinstance(ret, values.Object))

    ret_ = []
    if isinstance(ret.get_value(), values.TupleValue):
        ret_.extend([v.get_obj().get_value() for v in ret.get_value().values])
    elif isinstance(ret, list):
        ret_ = [r.get_value() for r in ret]
    else:
        ret_ = [ret.get_value()]

    for v in value_args:
        graph.add_input_value(v)

    for v in ret_:
        graph.add_output_value(v)

    return (value_args, ret_, graph)
