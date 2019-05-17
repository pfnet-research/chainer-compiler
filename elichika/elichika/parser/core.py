import chainer
import chainer as C
import chainer.functions as F
import chainer.links as L
import inspect
import weakref
import sys
from elichika.parser import config
from elichika.parser import nodes
from elichika.parser import vevaluator
from elichika.parser import values
from elichika.parser import links_builtin
from elichika.parser import functions
from elichika.parser import functions_builtin
from elichika.parser import functions_ndarray
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
        if links_builtin.is_builtin_chainer_link(i):
            return links_builtin.ChainerLinkInstance(m, i)

        if isinstance(i, chainer.ChainList):
            return links_builtin.ChainerChainListInstance(m, i)

        if isinstance(i, chainer.Link):
            return links_builtin.ChainerChainInstance(m, i)

        return None

    values.instance_converters.append(instance_converter)

    # generate default module
    default_module = values.Module(sys.modules[model.__module__])

    # chainer
    chainer_module_name = get_module_name(
        C, default_module.internal_module)

    if chainer_module_name != '':
        c_dict = values.ValueRef(values.ModuleValue())

        # a substitute of Variable
        c_variable = values.FuncValue(functions_ndarray.NDArrayFunction(), None)
        c_dict.get_field().get_attribute('Variable').revise(values.ValueRef(c_variable))

        default_module.set_default_value(chainer_module_name, c_dict)

    # chainer.functions
    chainer_functions_module_name = get_module_name(
        F, default_module.internal_module)

    if chainer_functions_module_name != '':
        f_dict = values.ValueRef(values.ModuleValue())

        def add_chainer_funtion(name:'str', func, ret_value_func = None):
            if ret_value_func is None:
                f = values.FuncValue(
                    functions_builtin.ChainerFunction(func), None)
            else:
                f = values.FuncValue(
                    functions_builtin.ChainerFunction(func, ret_value_func=ret_value_func), None)
            f_dict.get_field().get_attribute(name).revise(values.ValueRef(f))

            values.function_converters[func] = f

        def ret_tuple():
            return values.TupleValue()

        add_chainer_funtion('relu', F.relu)
        add_chainer_funtion('softmax', F.softmax)
        add_chainer_funtion('softmax_cross_entropy', F.softmax_cross_entropy)
        add_chainer_funtion('pad_sequence', F.pad_sequence)
        add_chainer_funtion('average_pooling_2d', F.average_pooling_2d)
        add_chainer_funtion('unpooling_2d', F.unpooling_2d)
        add_chainer_funtion('reshape', F.reshape)
        add_chainer_funtion('split_axis', F.split_axis, ret_value_func=ret_tuple)
        add_chainer_funtion('hstack', F.hstack)
        add_chainer_funtion('vstack', F.vstack)
        add_chainer_funtion('stack', F.stack)
        add_chainer_funtion('separate', F.separate, ret_value_func=ret_tuple)
        add_chainer_funtion('squeeze', F.squeeze)        
        add_chainer_funtion('reshape', F.reshape)
        add_chainer_funtion('swapaxes', F.swapaxes)
        add_chainer_funtion('dropout', F.dropout)
        add_chainer_funtion('concat', F.concat)
        add_chainer_funtion('matmul', F.matmul)
        add_chainer_funtion('max_pooling_2d', F.max_pooling_2d)
        add_chainer_funtion('resize_images', F.resize_images)
        add_chainer_funtion('tanh', F.tanh)
        add_chainer_funtion('sigmoid', F.sigmoid)
        add_chainer_funtion('broadcast_to', F.broadcast_to)
        add_chainer_funtion('expand_dims', F.expand_dims)
        add_chainer_funtion('local_response_normalization', F.local_response_normalization)

        if int(chainer.__version__[0]) >= 6:
            add_chainer_funtion('roi_max_pooling_2d', F.roi_max_pooling_2d)
            add_chainer_funtion('roi_average_pooling_2d', F.roi_average_pooling_2d)
            add_chainer_funtion('roi_max_align_2d', F.roi_max_align_2d)

        add_chainer_funtion('roi_average_align_2d', F.roi_average_align_2d)

        default_module.set_default_value(chainer_functions_module_name, f_dict)

    # numpy
    numpy_module_name = get_module_name(np, default_module.internal_module)
    if numpy_module_name != '':
        f_dict = values.ValueRef(values.ModuleValue())

        f_array = values.FuncValue(functions_ndarray.NDArrayFunction(), None)
        f_dict.get_field().get_attribute('array').revise(values.ValueRef(f_array))

        f_zeros = values.FuncValue(functions_ndarray.NDArrayZerosFunction(), None)
        f_dict.get_field().get_attribute('zeros').revise(values.ValueRef(f_zeros))

        f_full = values.FuncValue(functions_ndarray.NDArrayFullFunction(), None)
        f_dict.get_field().get_attribute('full').revise(values.ValueRef(f_full))

        f_ceil = values.FuncValue(functions_ndarray.NDArrayCeilFunction(), None)
        f_dict.get_field().get_attribute('ceil').revise(values.ValueRef(f_ceil))

        f_cumsum = values.FuncValue(functions_ndarray.NDArrayCumsumFunction(), None)
        f_dict.get_field().get_attribute('cumsum').revise(values.ValueRef(f_cumsum))

        f_dict.get_field().get_attribute('int32').revise(
            values.ValueRef(values.NumberValue(utils.numpy_type_2_int(np.int32))))
        f_dict.get_field().get_attribute('float32').revise(
            values.ValueRef(values.NumberValue(utils.numpy_type_2_int(np.float32))))

        default_module.set_default_value(numpy_module_name, f_dict)

    m_range = values.FuncValue(functions_builtin.RangeFunction(), None)
    default_module.set_default_value('range', values.ValueRef(m_range))

    m_list = values.FuncValue(functions_builtin.ListFunction(), None)
    default_module.set_default_value('list', values.ValueRef(m_list))

    m_len = values.FuncValue(functions_builtin.LenFunction(), None)
    default_module.set_default_value('len', values.ValueRef(m_len))

    model_inst = values.parse_instance(default_module, '', model)
    forward_func = model_inst.try_get_and_store_obj('forward', None)

    # convert args
    finput = functions.FunctionArgInput()

    value_args = []
    ind = 0

    node_input = nodes.NodeInput('input')

    for arg in args:
        varg = values.parse_instance(default_module, '', arg, None)
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

    node_input.set_outputs(value_args)

    graph = Graph()
    graph.root_graph = graph
    graph.add_node(node_input)

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

    if ret is None or isinstance(ret, values.NoneValue):
        if config.show_warnings:
            print('Failed to compile. output is None.')
        return (value_args, None, graph)

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
