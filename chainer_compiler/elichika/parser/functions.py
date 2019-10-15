
import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import ast
import gast
import weakref
from enum import Enum
import re
import numpy as np

from chainer_compiler.elichika.parser import vevaluator
from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser import core
from chainer_compiler.elichika.parser import config
from chainer_compiler.elichika.parser import canonicalizer


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
        copied.dtype = value.dtype
        return copied

    if isinstance(value, values.ListValue):
        copied = values.ListValue()
        copied.dtype = value.dtype
        copied.vtype = value.vtype
        if value.internal_value is not None:
            copied.internal_value = value.internal_value.copy()
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
        copied.dtype = value.dtype
        copied.vtype = value.vtype
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
    Dummy = 2,
    Input = 3,

def generate_value_with_type(value: 'values.Value', type_,  suffix_type = SuffixType.Unknown):
    assert(isinstance(value, values.Value))
    ret = None
    if isinstance(value, values.TensorValue):
        ret = values.TensorValue()
        ret.shape = value.shape
        ret.dtype = type_

    elif isinstance(value, values.NumberValue):
        ret = values.NumberValue(None)
        ret.dtype = type_

    else:
        assert(False)

    if suffix_type == SuffixType.Unknown:
        ret.name = value.name + '_st'
    elif suffix_type == SuffixType.Unused:
        ret.name = value.name + '_unused'
    elif suffix_type == SuffixType.Dummy:
        ret.name = value.name + '_dummy'
    elif suffix_type == SuffixType.Input:
        ret.name = value.name + '_in'
    else:
        assert(False)

    return ret


def generate_value_with_same_type(value: 'values.Value', is_dummy_value = False, suffix_type = SuffixType.Unknown):
    assert(isinstance(value, values.Value))
    ret = None
    if isinstance(value, values.TensorValue):
        ret = values.TensorValue()
        ret.shape = value.shape
        ret.dtype = value.dtype

    elif isinstance(value, values.NumberValue):
        dtype = None
        if value.internal_value is None:
            dtype = value.dtype
        elif isinstance(value.internal_value, int):
            dtype = np.array(value.internal_value).dtype
        elif isinstance(value.internal_value, float):
            dtype = np.array(value.internal_value).dtype

        ret = values.NumberValue(None)
        ret.dtype = dtype

    elif isinstance(value, values.StrValue):
        ret = values.StrValue(None)

    elif isinstance(value, values.BoolValue):
        ret = values.BoolValue(None)

    elif isinstance(value, values.ListValue):
        ret = values.ListValue(None)
        ret.dtype = value.dtype
        ret.vtype = value.vtype

    elif isinstance(value, values.NoneValue):
        ret = values.NoneValue()

    elif isinstance(value, values.TupleValue):
        ret = values.TupleValue()
        ret.dtype = value.dtype
        ret.vtype = value.vtype

    elif isinstance(value, values.RangeValue):
        ret = values.RangeValue()

    elif isinstance(value, values.UnknownValue):
        ret = values.UnknownValue()

    elif ret is None and isinstance(value, values.Value):
        ret = values.Value()

    else:
        assert(False)

    assert(ret is not None)

    ret.is_dummy_value = is_dummy_value
    if suffix_type == SuffixType.Unknown:
        ret.name = value.name + '_st'
    elif suffix_type == SuffixType.Unused:
        ret.name = value.name + '_unused'
    elif suffix_type == SuffixType.Dummy:
        ret.name = value.name + '_dummy'
    elif suffix_type == SuffixType.Input:
        ret.name = value.name + '_in'
    else:
        assert(False)

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
    def __init__(self, name: 'str' = '', obj: 'values.Object' = None):
        self.name = name
        self.obj = obj


class FunctionArgCollection():
    def __init__(self):
        self.args = {}  # Dict[str,FunctionArg]
        self.args_list = []

    def add_arg(self, name, value):

        if isinstance(value, values.Value):
            value = values.Object(value)

        assert not(name in self.args.keys())

        fa = FunctionArg(name, value)
        self.args_list.append(fa)
        self.args[fa.name] = fa

    def analyze_args(self, func):
        sig = inspect.signature(func)
        argspec = inspect.getfullargspec(func)

        parameter_count = 0
        for k, v in sig.parameters.items():
            # TODO improve it
            if k == 'kwargs':
                continue
            parameter_count += 1

        isSelfRemoved = parameter_count != len(argspec.args) + len(argspec.kwonlyargs)

        if isSelfRemoved:
            self.add_arg(argspec.args[0], None)

        for k, v in sig.parameters.items():
            # TODO improve it
            if k == 'kwargs':
                continue

            self.add_arg(v.name, values.parse_instance(None, v.name, v.default))

    def merge_inputs(self, self_Object, inputs: 'FunctionArgInput') -> 'FunctionArgInput':
        ret = FunctionArgInput()

        for fa in self.get_args():
            ret.inputs.append(fa.obj)
            ret.keywords[fa.name] = fa.obj

        inputs_ = inputs.inputs.copy()
        keywords_ = inputs.keywords.copy()

        if self_Object is not None:
            inputs_ = [self_Object] + inputs_
            keywords_[self.args_list[0].name] = self_Object

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

def auto_set_unset(func, flag):
    def decorated(self, *args, **kwargs):
        self.history.append((flag, self.flags[flag]))
        ret = func(self, *args, **kwargs)
        return ret
    return decorated

class StackTrace:
    def __init__(self):
        self.lineprops = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.lineprops.pop()
        return False

    def append(self, line):
        self.lineprops.append(line)

class VEvalContext:
    def __init__(self):
        self.stacktrace = StackTrace()
        self.history = []
        self.flags = {
            "eval_as_written_target": False,
            "ignore_branch": False,
            "for_unroll": False
        }
        self.flags_cache = []

        list(map(lambda flag: setattr(self.__class__, '_' + flag, property(fset=lambda obj, value: VEvalContext.generic_setter(obj, flag, value),
                                                                           fget=lambda obj: VEvalContext.generic_getter(obj, flag))),
                 self.flags.keys()))
        list(map(lambda flag: setattr(self.__class__, flag, auto_set_unset(lambda obj, default=True: VEvalContext.generic_setter(obj, flag, default), flag)),
                 self.flags.keys()))

    def generic_setter(self, name, value = True):
        self.flags[name] = value
        return self

    def generic_getter(self, name):
        return self.flags[name]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        flag, saved_value = self.history.pop()
        self.flags[flag] = saved_value
        return False


class FunctionBase():
    def __init__(self):
        self.name = ''
        self.is_property = False
        self.args = FunctionArgCollection()

        self.base_func = None

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'FunctionArgInput',
              context: 'VEvalContext' = None, line=-1):
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
        self.filename = inspect.getfile(func)

        sourcelines = inspect.getsourcelines(func)
        if sourcelines is None or len(sourcelines) < 1:
            utils.print_warning('Failed to parase {}'.format(classinfo), utils.LineProperty())
            return

        self.lineno = sourcelines[1]
        self.classinfo = classinfo

        original_code = inspect.getsource(func)
        code = utils.clip_head(original_code)

        self.args.analyze_args(func)

        ast_ = gast.ast_to_gast(ast.parse(code)).body[0]
        self.ast = canonicalizer.Canonicalizer().visit(ast_)

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'FunctionArgInput',
              context: 'VEvalContext' = None, line=-1):
        ret = values.Object(values.UserDefinedInstance(
            module, None, self.classinfo))
        inst = ret

        func_field = values.Field()
        func_field.set_module(module)

        # add args
        funcArgs = self.args.merge_inputs(inst, args)

        for k, v in funcArgs.keywords.items():
            func_field.get_field().get_attribute(k, from_module=False).revise(v)

        astc = vevaluator.AstContext(self.ast.body, self.lineno - 1, filename=self.filename)
        vevaluator.veval_ast(astc, func_field, graph, context)

        # dispose because of exit from function
        func_field.dispose()

        return ret


class UserDefinedFunction(FunctionBase):
    def __init__(self, func):
        super().__init__()

        self.inst = func
        self.name = func.__name__
        self.filename = inspect.getfile(func)
        sourcelines = inspect.getsourcelines(func)
        self.lineno = sourcelines[1]
        self.args.analyze_args(func)

        if (func.__name__ == (lambda: None).__name__):
            original_code = utils.lambda_source(func)
            code = 'return ' + original_code[re.search('lambda.*?:', original_code).end():]
            self.ast = gast.ast_to_gast(ast.parse(code))
        else:
            original_code = inspect.getsource(func)
            code = utils.clip_head(original_code)
            ast_ = gast.ast_to_gast(ast.parse(code)).body[0]
            self.ast = canonicalizer.Canonicalizer().visit(ast_)

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'FunctionArgInput',
              context: 'VEvalContext' = None, line=-1):

        if context is None:
            context = VEvalContext()

        context.stacktrace.append(line)
        with context.stacktrace:

            func_field = values.Field()
            func_field.set_module(module)

            # add args
            funcArgs = self.args.merge_inputs(inst, args)

            for k, v in funcArgs.keywords.items():
                func_field.get_field().get_attribute(k, from_module=False).revise(utils.try_get_obj(v, self.name, utils.LineProperty()))

            astc = vevaluator.AstContext(self.ast.body, self.lineno - 1, filename=self.filename)
            ret = vevaluator.veval_ast(astc, func_field, graph, context)

            # dispose because of exit from function
            func_field.dispose()

            return ret


class UnimplementedFunction(FunctionBase):
    def __init__(self, func):
        super().__init__()

        if isinstance(func, str):
            self.name = func
        else:
            self.name = func.__name__

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'FunctionArgInput',
              context: 'VEvalContext' = None, line=-1):
        raise utils.UnimplementedError('{} is unimplemented.'.format(self.name), utils.LineProperty(line))


class UserDefinedFunctionFromAst(FunctionBase):
    def __init__(self, astc, args, func_field):
        super().__init__()
        assert isinstance(astc.nast, (gast.FunctionDef, gast.Lambda))

        self.name = astc.gast.name if isinstance(astc.nast, gast.FunctionDef) else (lambda: None).__name__
        self.args = args
        self.func_field = func_field
        if isinstance(astc.nast, gast.Lambda):
            astc.nast.body = gast.Return(value=astc.nast.body) # Add return to the body
        self.ast = astc.nast
        self.filename = astc.filename
        self.lineno = astc.lineno

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'FunctionArgInput',
              context: 'VEvalContext' = None, line=-1):
        self.func_field.set_module(module)

        # add args
        funcArgs = self.args.merge_inputs(inst, args)

        for k, v in funcArgs.keywords.items():
            self.func_field.get_field().get_attribute(k, from_module=False).revise(utils.try_get_obj(v, self.name, utils.LineProperty()))

        astc = vevaluator.AstContext(self.ast.body, self.lineno - 1, filename=self.filename)
        ret = vevaluator.veval_ast(astc, self.func_field, graph, context)

        # dispose because of exit from function
        self.func_field.dispose()

        return ret

class CheckAttributeValueFunction(FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'check_attribute_value'
        self.args.add_arg('actual_value', values.NoneValue())
        self.args.add_arg('expected_value', values.NoneValue())
        self.args.add_arg('func_name', values.NoneValue())
        self.args.add_arg('arg_name', values.NoneValue())

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'FunctionArgInput',
              context: 'VEvalContext' = None, line=-1):

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        if type(vargs[0]) != type(vargs[1]) or vargs[0].get_constant_value() != vargs[1].get_constant_value():
            raise Exception("Value must be {} : {} from {} in {}".format(
                vargs[1].get_constant_value(),
                vargs[3].get_constant_value(),
                vargs[2].get_constant_value(),
                context.stacktrace.lineprops[-1]))

class CheckAttributeScalarFunction(FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'check_attribute_scalar'
        self.args.add_arg('value', values.NoneValue())
        self.args.add_arg('func_name', values.NoneValue())
        self.args.add_arg('arg_name', values.NoneValue())

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'FunctionArgInput',
                      context: 'VEvalContext' = None, line=-1):

        funcArgs = self.args.merge_inputs(inst ,args)
        vargs = funcArgs.get_value().inputs

        if not isinstance(vargs[0], values.NumberValue):
            raise Exception("A number is only supported {} : from {} in {}".format(
                vargs[2].get_constant_value(),
                vargs[1].get_constant_value(),
                context.stacktrace.lineprops[-1]))
