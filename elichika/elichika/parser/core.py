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


def convert_model(model: 'chainer.Chain', args=[]):
    # reset values
    values.reset_field_and_attributes()
    utils.reset_guid()

    values.instance_converters.clear()

    def instance_converter(m, i):
        if values_builtin.is_builtin_chainer_link(i):
            return values_builtin.ChainerLinkInstance(m, i)
        return None

    values.instance_converters.append(instance_converter)

    # generate default module
    default_module = values.Module(sys.modules[model.__module__])

    # chainer.functions
    chainer_functions_module_name = get_module_name(
        F, default_module.internal_module)

    if chainer_functions_module_name != '':
        f_dict = values.ValueRef(values.ModuleValue())
        f_relu = values.FuncValue(
            functions_builtin.ChainerFunction(F.relu), None)
        f_dict.get_field().get_attribute('relu').revise(values.ValueRef(f_relu))
        f_softmax = values.FuncValue(
            functions_builtin.ChainerFunction(F.softmax), None)
        f_dict.get_field().get_attribute('softmax').revise(values.ValueRef(f_softmax))
        f_softmax_cross_entropy = values.FuncValue(
            functions_builtin.ChainerFunction(F.softmax_cross_entropy), None)
        f_dict.get_field().get_attribute('softmax_cross_entropy').revise(
            values.ValueRef(f_softmax_cross_entropy))
        f_pad_sequence = values.FuncValue(
            functions_builtin.ChainerFunction(F.pad_sequence), None)
        f_dict.get_field().get_attribute('pad_sequence').revise(
            values.ValueRef(f_pad_sequence))
        f_average_pooling_2d = values.FuncValue(
            functions_builtin.ChainerFunction(F.average_pooling_2d), None)
        f_dict.get_field().get_attribute('average_pooling_2d').revise(values.ValueRef(f_average_pooling_2d))
        default_module.set_default_value(chainer_functions_module_name, f_dict)

    # numpy
    numpy_module_name = get_module_name(np, default_module.internal_module)
    if numpy_module_name != '':
        f_dict = values.ValueRef(values.ModuleValue())

        f_array = values.FuncValue(functions_builtin.NDArrayFunction(), None)
        f_dict.get_field().get_attribute('array').revise(values.ValueRef(f_array))

        f_dict.get_field().get_attribute('int32').revise(
            values.ValueRef(values.NumberValue(utils.numpy_type_2_int(np.int32))))
        f_dict.get_field().get_attribute('float32').revise(
            values.ValueRef(values.NumberValue(utils.numpy_type_2_int(np.float32))))

        default_module.set_default_value(numpy_module_name, f_dict)

    m_range = values.FuncValue(functions_builtin.RangeFunction(), None)
    default_module.set_default_value('range', values.ValueRef(m_range))

    m_list = values.FuncValue(functions_builtin.ListFunction(), None)
    default_module.set_default_value('list', values.ValueRef(m_list))

    model_inst = values.parse_instance(default_module, '', model)
    forward_func = model_inst.try_get_and_store_obj('forward')

    # convert args
    finput = functions.FunctionArgInput()

    value_args = []
    ind = 0
    for arg in args:
        varg = values.parse_instance(default_module, '', arg, None, True)
        varg.name = 'in_' + str(ind)
        varg.get_value().name = 'in_' + str(ind)

        # make value unknown
        # if isinstance(varg.get_value(), values.TupleValue):
        #    for i in range(len(varg.get_value().internal_value)):
        #        varg.get_value().internal_value[i] = None
        # else:
        varg.get_value().internal_value = None

        finput.inputs.append(varg)
        value_args.append(varg.get_value())
        ind += 1

    graph = Graph()
    forward_func_value = forward_func.get_value()
    ret = forward_func_value.func.vcall(
        default_module, graph, forward_func_value.obj, finput)
    assert(ret is None or isinstance(ret, values.ValueRef))

    def try_get_value(value) -> 'values.Value':
        if isinstance(value, values.Value):
            return value

        if isinstance(value, values.ValueRef):
            return value.get_value()

        if isinstance(value, values.Attribute):
            return value.get_ref().get_value()

    ret_ = []
    if isinstance(ret.get_value(), values.TupleValue):
        if ret.get_value().internal_value is not None:
            for v in ret.get_value().internal_value:
                assert(v is not None)
                ret_.append(try_get_value(v))
        else:
            ret_ = [ret.get_value()]

    elif isinstance(ret, list):
        ret_ = [r.get_value() for r in ret]
    else:
        ret_ = [ret.get_value()]

    for v in value_args:
        graph.add_input_value(v)

    for v in ret_:
        graph.add_output_value(v)

    return (value_args, ret_, graph)
