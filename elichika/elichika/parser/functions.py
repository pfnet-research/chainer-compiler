
import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import ast
import gast
import weakref
from enum import Enum

import numpy as np

from elichika.parser import vevaluator
from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import utils
from elichika.parser import core
from elichika.parser import config


def generate_copied_value(value: 'values.Value'):
    assert(isinstance(value, values.Value))

    if isinstance(value, values.NumberValue):
        copied = values.NumberValue(value.internal_value)
        copied.dtype = value.dtype
        return copied

    if isinstance(value, values.TensorValue):
        copied = values.TensorValue()
        copied.value = value.value
        copied.shape = value.shape
        return copied

    if isinstance(value, values.ListValue):
        copied = values.ListValue()
        copied.is_any = value.is_any
        copied.values = value.values.copy()
        return copied

    if isinstance(value, values.NoneValue):
        copied = values.NoneValue()
        return copied

    if isinstance(value, values.BoolValue):
        copied = values.BoolValue(value.internal_value)
        return copied

    if isinstance(value, values.StrValue):
        copied = values.StrValue(value.internal_value)
        return copied

    if isinstance(value, values.RangeValue):
        copied = values.RangeValue()
        return copied

    if isinstance(value, values.TupleValue):
        if value.internal_value is not None:
            copied = values.TupleValue(value.internal_value.copy())
        else:
            copied = values.TupleValue(value.internal_value)
        return copied

    if config.show_warnings:
        print('Unknown type {} is copied'.format(value))

    return values.Value()


def generate_tensor_value_with_undefined_shape_size(value: 'values.TensorValue'):
    assert(isinstance(value, values.TensorValue))
    ret = values.TensorValue()
    ret.shape = tuple([-1 for v in value.shape])
    return ret


class SuffixType(Enum):
    Unknown = 0,
    Unused = 1,

def generate_value_with_same_type(value: 'values.Value', has_default = False, suffix_type = SuffixType.Unknown):
    assert(isinstance(value, values.Value))
    ret = None
    if isinstance(value, values.TensorValue):
        ret = values.TensorValue()
        ret.shape = value.shape
        ret.dtype = value.dtype

    if isinstance(value, values.NumberValue):
        dtype = None
        if value.internal_value is None:
            dtype = value.dtype
        elif isinstance(value.internal_value, int):
            dtype = np.array(value.internal_value).dtype
        elif isinstance(value.internal_value, float):
            dtype = np.array(value.internal_value).dtype

        if has_default:
            if dtype == np.array(0).dtype:
                ret = values.NumberValue(0)
            elif dtype == np.array(0.0).dtype:
                ret = values.NumberValue(0.0)
            else:
                ret = values.NumberValue(None)
        else:
            ret = values.NumberValue(None)
        ret.dtype = dtype

    if isinstance(value, values.StrValue):
        if has_default:
            ret = values.StrValue('')
        else:
            ret = values.StrValue(None)
            
    if isinstance(value, values.BoolValue):
        if has_default:
            ret = values.BoolValue(False)
        else:
            ret = values.BoolValue(None)

    if isinstance(value, values.ListValue):
        ret = values.ListValue(None)

    if isinstance(value, values.NoneValue):
        ret = values.NoneValue()

    if isinstance(value, values.TupleValue):
        ret = values.TupleValue()

    if isinstance(value, values.UnknownValue):
        ret = values.UnknownValue()
        if has_default:
            ret.internal_value = 0

    if ret is None and isinstance(value, values.Value):
        ret = values.Value()

    if ret is not None:
        if suffix_type == SuffixType.Unknown:
            ret.name = value.name + '_st'
        if suffix_type == SuffixType.Unused:
            ret.name = value.name + '_unused'
    return ret


class FunctionArgInput():
    def __init__(self):
        self.inputs = []
        self.keywords = {}

    def get_value(self) -> "FunctionArgValueInput":
        ret = functions.FunctionArgValueInput()
        ret.inputs = [v.get_value() for v in self.inputs]

        keywords_ = {}
        for k, v in self.keywords.items():
            keywords_[k] = v.get_value()
        ret.keywords = keywords_
        return ret

class FunctionArgValueInput():
    def __init__(self):
        self.inputs = [] # List[values.Value]
        self.keywords = {}  # Dict[str,values.Value]

    def get_value(self, key) -> 'values.Value':
        if isinstance(key, int):
            return self.inputs[key]
        if isinstance(key, str) and key in self.keywords.keys():
            return self.keywords[key]
        return None


class FunctionArg():
    def __init__(self, name: 'str' = '', obj: 'values.ValueRef' = None):
        self.name = name
        self.obj = obj


class FunctionArgCollection():
    def __init__(self):
        self.args = {}  # Dict[str,FunctionArg]
        self.args_list = []
        
    def add_arg(self, name, value):

        if isinstance(value, values.Value):
            value = values.ValueRef(value)

        fa = FunctionArg(name, value)
        self.args_list.append(fa)
        self.args[fa.name] = fa

    def analyze_args(self, func):
        sig = inspect.signature(func)
        argspec = inspect.getargspec(func)

        isSelfRemoved = len(sig.parameters.keys()) != len(argspec[0])

        if isSelfRemoved:
            self.add_arg(argspec[0][0], None)

        for k, v in sig.parameters.items():
            self.add_arg(v.name, values.parse_instance(None, v.name, v.default))

    def merge_inputs(self, self_valueref, inputs: 'FunctionArgInput') -> 'FunctionArgInput':
        ret = FunctionArgInput()
        
        for fa in self.get_args():
            ret.inputs.append(fa.obj)
            ret.keywords[fa.name] = fa.obj

        inputs_ = inputs.inputs.copy()
        keywords_ = inputs.keywords.copy()

        if self_valueref is not None:
            inputs_ = [self_valueref] + inputs_
            keywords_[self.args_list[0].name] = self_valueref

        for i in range(len(inputs_)):
            ret.inputs[i] = inputs_[i]
            ret.keywords[self.args_list[i].name] = ret.inputs[i]

        for k, v in keywords_.items():
            if k in ret.keywords.keys():
                ret.keywords[k] = v

            for i in range(len(self.args_list)):
                if self.args_list[i].name == k:
                    ret.inputs[i] = v

        return ret

    def get_value(self, key) -> 'values.Value':
        if isinstance(key, int):
            return self.args_list[key].obj.get_value()
        if isinstance(key, str) and key in self.args.keys():
            return self.args[key].obj.get_value()
        return None

    def get_values(self) -> 'List[values.Value]':
        return [a.obj.get_value() for a in self.args_list]

    def get_args(self) -> 'List[FunctionArg]':
        ret = []

        for fa in self.args_list:
            ret.append(FunctionArg(fa.name, fa.obj))
        return ret


class FunctionBase():
    def __init__(self):
        self.name = ''
        self.is_property = False
        self.args = FunctionArgCollection()

        self.base_func = None

    def vcall(self, module: 'values.Field', graph: 'core.Graph', inst: 'values.Value', args=[], line=-1):
        return None


class UserDefinedClassConstructorFunction(FunctionBase):
    def __init__(self, classinfo):
        super().__init__()

        members = inspect.getmembers(classinfo)
        init_func = [m[1] for m in members if m[0] == '__init__']
        assert(len(init_func) == 1)

        func = init_func[0]
        self.inst = func
        self.name = func.__name__
        self.lineno = inspect.getsourcelines(func)[1]
        self.classinfo = classinfo

        original_code = inspect.getsource(func)
        code = utils.clip_head(original_code)

        self.args.analyze_args(func)

        self.ast = gast.ast_to_gast(ast.parse(code)).body[0]

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.ValueRef', args: 'FunctionArgInput', line=-1):
        ret = values.ValueRef(values.UserDefinedInstance(
            module, None, self.classinfo))
        inst = ret

        func_field = values.Field()
        func_field.set_module(module)

        # add args
        funcArgs = self.args.merge_inputs(inst, args)

        for k, v in funcArgs.keywords.items():
            func_field.get_field().get_attribute(k).revise(v)

        astc = vevaluator.AstContext(self.ast.body, self.lineno - 1)
        vevaluator.veval_ast(astc, func_field, graph)

        return ret


class UserDefinedFunction(FunctionBase):
    def __init__(self, func):
        super().__init__()

        self.inst = func
        self.name = func.__name__
        self.lineno = inspect.getsourcelines(func)[1]

        code = utils.clip_head(inspect.getsource(func))

        self.args.analyze_args(func)

        self.ast = gast.ast_to_gast(ast.parse(code)).body[0]

    def vcall(self, module: 'values.Field', graph: 'core.Graph', inst: 'values.ValueRef', args: 'FunctionArgInput', line=-1):
        func_field = values.Field()
        func_field.set_module(module)

        # add args
        funcArgs = self.args.merge_inputs(inst, args)

        for k, v in funcArgs.keywords.items():
            func_field.get_field().get_attribute(k).revise(v)

        astc = vevaluator.AstContext(self.ast.body, self.lineno - 1)
        return vevaluator.veval_ast(astc, func_field, graph)
