import chainer
import chainer as C
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
import inspect
import weakref
import sys
from chainer_compiler.elichika.parser import config
from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import vevaluator
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import links_builtin
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import functions_builtin
from chainer_compiler.elichika.parser import functions_ndarray
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser.graphs import Graph
from chainer_compiler.elichika.parser import flags
import numpy as np
import six

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

    values.function_converters.clear()
    values.builtin_function_converters.clear()
    values.instance_converters.clear()

    def instance_converter(m, i):
        if links_builtin.is_builtin_chainer_link(i):
            return links_builtin.ChainerLinkInstance(m, i)

        if isinstance(i, chainer.ChainList):    
            module = values.ValueRef(values.ModuleValue(sys.modules[i.__module__]))
            return links_builtin.ChainerChainListInstance(module, i)

        if isinstance(i, chainer.Link):
            module = values.ValueRef(values.ModuleValue(sys.modules[i.__module__]))
            return links_builtin.ChainerChainInstance(module, i)

        return None

    values.instance_converters.append(instance_converter)

    # chainer
    c_variable = values.FuncValue(functions_ndarray.NDArrayFunction(), None)
    values.function_converters[chainer.Variable] = c_variable

    # chainer.functions
    def add_chainer_function(name:'str', func, ret_value_func = None):
        if ret_value_func is None:
            f = values.FuncValue(
                functions_builtin.ChainerFunction(func), None)
        else:
            f = values.FuncValue(
                functions_builtin.ChainerFunction(func, ret_value_func=ret_value_func), None)

        values.function_converters[func] = f

    def ret_tuple():
        ret = values.TupleValue()
        ret.vtype = values.TensorValue
        return ret

    add_chainer_function('relu', F.relu)
    add_chainer_function('softmax', F.softmax)
    add_chainer_function('softmax_cross_entropy', F.softmax_cross_entropy)
    add_chainer_function('pad_sequence', F.pad_sequence)
    add_chainer_function('average_pooling_2d', F.average_pooling_2d)
    add_chainer_function('unpooling_2d', F.unpooling_2d)
    add_chainer_function('reshape', F.reshape)
    add_chainer_function('split_axis', F.split_axis, ret_value_func=ret_tuple)
    add_chainer_function('hstack', F.hstack)
    add_chainer_function('vstack', F.vstack)
    add_chainer_function('stack', F.stack)
    add_chainer_function('separate', F.separate, ret_value_func=ret_tuple)
    add_chainer_function('squeeze', F.squeeze)
    add_chainer_function('swapaxes', F.swapaxes)
    add_chainer_function('dropout', F.dropout)
    add_chainer_function('concat', F.concat)
    add_chainer_function('matmul', F.matmul)
    add_chainer_function('max_pooling_2d', F.max_pooling_2d)
    add_chainer_function('resize_images', F.resize_images)
    add_chainer_function('tanh', F.tanh)
    add_chainer_function('sigmoid', F.sigmoid)
    add_chainer_function('broadcast_to', F.broadcast_to)
    add_chainer_function('expand_dims', F.expand_dims)
    add_chainer_function('local_response_normalization', F.local_response_normalization)
    add_chainer_function('mean', F.mean)
    add_chainer_function('average', F.average)
    add_chainer_function('sum', F.sum)

    if int(chainer.__version__[0]) >= 6:
        add_chainer_function('roi_max_pooling_2d', F.roi_max_pooling_2d)
        add_chainer_function('roi_average_pooling_2d', F.roi_average_pooling_2d)
        add_chainer_function('roi_max_align_2d', F.roi_max_align_2d)

    add_chainer_function('roi_average_align_2d', F.roi_average_align_2d)

    # numpy
    f_array = values.FuncValue(functions_ndarray.NDArrayFunction(), None)
    f_zeros = values.FuncValue(functions_ndarray.NDArrayZerosFunction(), None)
    f_full = values.FuncValue(functions_ndarray.NDArrayFullFunction(), None)
    f_ceil = values.FuncValue(functions_ndarray.NDArrayCeilFunction(), None)
    f_cumsum = values.FuncValue(functions_ndarray.NDArrayCumsumFunction(), None)

    f_int32 = values.FuncValue(functions_ndarray.NDArrayInt32(), None)
    f_float32 = values.FuncValue(functions_ndarray.NDArrayFloat32(), None)

    values.function_converters[np.array] = f_array
    values.function_converters[np.zeros] = f_zeros
    values.function_converters[np.full] = f_full
    values.function_converters[np.ceil] = f_ceil
    values.function_converters[np.cumsum] = f_cumsum
    values.function_converters[np.int32] = f_int32
    values.function_converters[np.float32] = f_float32

    m_range = values.FuncValue(functions_builtin.RangeFunction(), None)
    values.builtin_function_converters['range'] = m_range

    m_len = values.FuncValue(functions_builtin.LenFunction(), None)
    values.builtin_function_converters['len'] = m_len

    values.function_converters[six.moves.range] = m_range

    m_list = values.FuncValue(functions_builtin.ListFunction(), None)
    values.builtin_function_converters['list'] = m_list

    m_print = values.FuncValue(functions_builtin.PrintFunction(), None)
    values.builtin_function_converters['print'] = m_print

    m_getattr = values.FuncValue(functions_builtin.GetAttrFunction(), None)
    values.builtin_function_converters['getattr'] = m_getattr

    m_to_gpu = values.FuncValue(functions_builtin.CopyFunction(cuda.to_gpu), None)
    values.function_converters[cuda.to_gpu] = m_to_gpu

    m_to_cpu = values.FuncValue(functions_builtin.CopyFunction(cuda.to_cpu), None)
    values.function_converters[cuda.to_cpu] = m_to_cpu

    # generate VEvalFlag functions
    def add_veval_flag_function(name:'str', func):
        f = values.FuncValue(functions_builtin.VEvalOptionFunction(func), None)
        values.builtin_function_converters[name] = f

    add_veval_flag_function('eval_as_written_target', flags.eval_as_written_target)
    add_veval_flag_function('ignore_branch', flags.ignore_branch)
    add_veval_flag_function('for_unroll', flags.for_unroll)

    # generate default module
    default_module = values.ValueRef(values.ModuleValue(sys.modules[model.__module__]))

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
        default_module, graph, forward_func_value.obj, finput).get_ref()
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
