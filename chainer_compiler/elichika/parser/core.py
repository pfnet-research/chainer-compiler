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
from chainer_compiler.elichika.parser import functions_list
from chainer_compiler.elichika.parser import functions_dict
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser.graphs import Graph
from chainer_compiler.elichika.parser import flags
from chainer_compiler.elichika.parser import custom_functions
from chainer_compiler.elichika.parser import functions_onnx

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
            module = values.Object(values.ModuleValue(sys.modules[i.__module__]))
            return links_builtin.ChainerChainListInstance(module, i)

        if isinstance(i, chainer.Link):
            module = values.Object(values.ModuleValue(sys.modules[i.__module__]))
            return links_builtin.ChainerChainInstance(module, i)

        return None

    values.instance_converters.append(instance_converter)

    custom_functions_module = values.Object(values.ModuleValue(custom_functions))

    # onnx
    functions_onnx_module = values.Object(values.ModuleValue(functions_onnx))
    def ret_same(funcArgs):
        return functions.generate_value_with_same_type(funcArgs.keywords['x'].get_value())

    values.function_converters[functions_onnx.onnx_abs] = values.FuncValue(functions_builtin.ChainerFunction(functions_onnx.onnx_abs, ret_value_func=ret_same), None, module=functions_onnx_module)

    # chainer
    c_variable = values.FuncValue(functions_ndarray.NDArrayFunction(), None)
    values.function_converters[chainer.Variable] = c_variable

    # chainer.functions
    def add_chainer_function(func, ret_value_func = None):
        if ret_value_func is None:
            f = values.FuncValue(
                functions_builtin.ChainerFunction(func), None)
        else:
            f = values.FuncValue(
                functions_builtin.ChainerFunction(func, ret_value_func=ret_value_func), None)

        values.function_converters[func] = f

    def ret_tuple(funcArgs = None):
        ret = values.TupleValue()
        ret.vtype = values.TensorValue
        return ret

    # register unsupported functions to show error when unsupported functions are called
    for f in F.__dict__.items():
        if inspect.isfunction(f[1]):
            values.function_converters[f[1]] = values.FuncValue(functions.UnimplementedFunction(f[1]), None)

    # activation
    add_chainer_function(F.elu)
    add_chainer_function(F.leaky_relu)
    add_chainer_function(F.log_softmax)
    add_chainer_function(F.relu)
    add_chainer_function(F.selu)
    add_chainer_function(F.sigmoid)
    add_chainer_function(F.softmax)
    add_chainer_function(F.tanh)

    add_chainer_function(F.softmax_cross_entropy)
    add_chainer_function(F.pad_sequence)
    add_chainer_function(F.average_pooling_2d)
    add_chainer_function(F.unpooling_2d)
    add_chainer_function(F.reshape)
    add_chainer_function(F.transpose)
    add_chainer_function(F.split_axis, ret_value_func=ret_tuple)
    add_chainer_function(F.hstack)
    add_chainer_function(F.vstack)
    add_chainer_function(F.stack)
    add_chainer_function(F.separate, ret_value_func=ret_tuple)
    add_chainer_function(F.squeeze)
    add_chainer_function(F.swapaxes)
    add_chainer_function(F.dropout)
    add_chainer_function(F.concat)
    add_chainer_function(F.matmul)
    add_chainer_function(F.max_pooling_2d)
    add_chainer_function(F.resize_images)
    add_chainer_function(F.broadcast_to)
    add_chainer_function(F.expand_dims)
    add_chainer_function(F.local_response_normalization)
    add_chainer_function(F.mean)
    add_chainer_function(F.average)
    add_chainer_function(F.sum)
    add_chainer_function(F.maximum)
    add_chainer_function(F.minimum)
    add_chainer_function(F.max)
    add_chainer_function(F.min)

    values.function_converters[F.absolute] = values.FuncValue(functions.UserDefinedFunction(custom_functions.chainer_absolute), None, module=custom_functions_module)

    add_chainer_function(F.sin)
    add_chainer_function(F.sinh)
    add_chainer_function(F.sign)
    add_chainer_function(F.cos)
    add_chainer_function(F.cosh)
    add_chainer_function(F.tan)
    add_chainer_function(F.tanh)
    add_chainer_function(F.arcsin)
    add_chainer_function(F.arccos)
    add_chainer_function(F.arctan)
    add_chainer_function(F.exp)
    add_chainer_function(F.log)
    add_chainer_function(F.sqrt)

    add_chainer_function(F.clip)

    values.function_converters[F.argmax] = values.FuncValue(functions_builtin.ChainerArgminmaxFunction(F.argmax), None)
    values.function_converters[F.argmin] = values.FuncValue(functions_builtin.ChainerArgminmaxFunction(F.argmin), None)

    values.function_converters[F.clipped_relu] = values.FuncValue(functions.UserDefinedFunction(custom_functions.chainer_clipped_relu), None, module=custom_functions_module)

    if int(chainer.__version__[0]) >= 6:
        add_chainer_function(F.roi_max_pooling_2d)
        add_chainer_function(F.roi_average_pooling_2d)
        add_chainer_function(F.roi_max_align_2d)

    add_chainer_function(F.roi_average_align_2d)

    # numpy
    f_array = values.FuncValue(functions_ndarray.NDArrayFunction(), None)
    f_zeros = values.FuncValue(functions_ndarray.NDArrayZerosFunction(), None)
    f_full = values.FuncValue(functions_ndarray.NDArrayFullFunction(), None)
    f_ceil = values.FuncValue(functions_ndarray.NDArrayCeilFunction(), None)
    f_cumsum = values.FuncValue(functions_ndarray.NDArrayCumsumFunction(), None)
    f_maximum = values.FuncValue(functions_ndarray.NDArrayChainerFunction(functions_ndarray.dummy_maximum), None)
    f_minimum = values.FuncValue(functions_ndarray.NDArrayChainerFunction(functions_ndarray.dummy_minimum), None)
    f_argmax = values.FuncValue(functions_ndarray.NDarrayArgminmaxFunction(functions_ndarray.dummy_argmax), None)
    f_argmin = values.FuncValue(functions_ndarray.NDarrayArgminmaxFunction(functions_ndarray.dummy_argmin), None)
    f_round = values.FuncValue(functions_ndarray.NDarrayRoundFunction(functions_ndarray.dummy_round), None)
    f_sqrt = values.FuncValue(functions_ndarray.NDarraySqrtFunction(functions_ndarray.dummy_sqrt), None)
    f_stack = values.FuncValue(functions_ndarray.NDarrayStackFunction(functions_ndarray.dummy_stack), None)
    f_reshape = values.FuncValue(functions_ndarray.NDarrayReshapeFunction(functions_ndarray.dummy_reshape), None)
    f_transpose = values.FuncValue(functions_ndarray.NDarrayTransposeFunction(functions_ndarray.dummy_transpose), None)

    f_int32 = values.FuncValue(functions_ndarray.NDArrayInt32(), None)
    f_float32 = values.FuncValue(functions_ndarray.NDArrayFloat32(), None)

    values.function_converters[np.array] = f_array
    values.function_converters[np.zeros] = f_zeros
    values.function_converters[np.full] = f_full
    values.function_converters[np.ceil] = f_ceil
    values.function_converters[np.cumsum] = f_cumsum
    values.function_converters[np.int32] = f_int32
    values.function_converters[np.float32] = f_float32
    values.function_converters[np.maximum] = f_maximum
    values.function_converters[np.minimum] = f_minimum
    values.function_converters[np.argmax] = f_argmax
    values.function_converters[np.argmin] = f_argmin
    values.function_converters[np.round] = f_round
    values.function_converters[np.sqrt] = f_sqrt
    values.function_converters[np.stack] = f_stack
    values.function_converters[np.reshape] = f_reshape
    values.function_converters[np.transpose] = f_transpose

    values.function_converters[np.clip] = values.FuncValue(functions.UserDefinedFunction(custom_functions.numpy_clip), None, module=custom_functions_module)
    values.function_converters[np.absolute] = values.FuncValue(functions.UserDefinedFunction(custom_functions.numpy_absolute), None, module=custom_functions_module)

    values.function_converters[custom_functions.check_attribute_value] = values.FuncValue(functions.CheckAttributeValueFunction(), None, module=custom_functions_module)

    values.function_converters[custom_functions.check_attribute_scalar] = values.FuncValue(functions.CheckAttributeScalarFunction(), None, module=custom_functions_module)

    values.builtin_function_converters['abs'] = values.FuncValue(functions.UserDefinedFunction(custom_functions.builtin_absolute), None, module=custom_functions_module)

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

    m_hasattr = values.FuncValue(functions_builtin.HasAttrFunction(), None)
    values.builtin_function_converters['hasattr'] = m_hasattr

    m_to_gpu = values.FuncValue(functions_builtin.CopyFunction(cuda.to_gpu), None)
    values.function_converters[cuda.to_gpu] = m_to_gpu

    m_to_cpu = values.FuncValue(functions_builtin.CopyFunction(cuda.to_cpu), None)
    values.function_converters[cuda.to_cpu] = m_to_cpu

    # generate VEvalFlag functions
    def add_veval_flag_function(name:'str', func):
        f = values.FuncValue(functions_builtin.VEvalContextFunction(func), None)
        values.builtin_function_converters[name] = f

    add_veval_flag_function('eval_as_written_target', flags.eval_as_written_target)
    add_veval_flag_function('ignore_branch', flags.ignore_branch)
    add_veval_flag_function('for_unroll', flags.for_unroll)

    # generate default module
    default_module = values.Object(values.ModuleValue(sys.modules[model.__module__]))

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
        default_module, graph, forward_func_value.obj, finput).get_obj()
    assert(ret is None or isinstance(ret, values.Object))

    def try_get_value(value) -> 'values.Value':
        if isinstance(value, values.Value):
            return value

        if isinstance(value, values.Object):
            return value.get_value()

        if isinstance(value, values.Attribute):
            return value.get_obj().get_value()

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
