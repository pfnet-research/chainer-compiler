
import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import ast, gast
import weakref

from elichika.parser import vevaluator
from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import utils
from elichika.parser import core
from elichika.parser import config

def generate_copied_value(value : 'values.Value'):
    if isinstance(value, values.NumberValue):
        copied = values.NumberValue(value.internal_value)
        return copied

    if isinstance(value, values.TensorValue):
        copied = values.TensorValue()
        copied.shape = value.shape
        return copied

    if isinstance(value, values.NoneValue):
        copied = values.NoneValue()
        return copied

    if config.show_warnings:
        print('Warning : Unimplemented copied_value {}'.format(value))

    return values.Value()

def generate_tensor_value_with_undefined_shape_size(value : 'values.TensorValue'):
    assert(isinstance(value, values.TensorValue))
    ret = values.TensorValue()
    ret.shape = tuple([-1 for v in value.shape])
    return ret


def generate_value_with_same_type(value : 'values.Value'):
    if isinstance(value, values.TensorValue):
        ret = values.TensorValue()
        ret.shape = value.shape
        return ret

    if isinstance(value, values.NumberValue):
        ret = values.NumberValue(None)
        return ret

    if isinstance(value, values.StrValue):
        ret = values.StrValue(None)
        return ret

    if isinstance(value, values.BoolValue):
        ret = values.BoolValue(None)
        return ret

    if isinstance(value, values.ListValue):
        ret = values.ListValue(None)
        return ret

    if isinstance(value, values.NoneValue):
        ret = values.NoneValue()
        return ret

    return None

class FunctionArg():
    def __init__(self):
        self.name = ''
        self.value = None

class FunctionBase():
    def __init__(self):
        self.name = ''
        self.funcArgs = []

    def parse_args(self, args):
        funcArgs = self.funcArgs.copy()

        for i in range(min(len(funcArgs), len(args))):
            if(args[i].name == ''):
                funcArgs[i].value = args[i].value

        for arg in args:
            if(arg.name != ''):
                for funcArg in funcArgs:
                    if funcArg.name == arg.name:
                        funcArg.value = arg.value
                        break

        return funcArgs

    def analyze_args(self, func):
        sig = inspect.signature(func)
        argspec = inspect.getargspec(func)

        isSelfRemoved = len(sig.parameters.keys()) != len(argspec[0])

        if isSelfRemoved:
            fa = FunctionArg()
            fa.name = argspec[0][0]
            fa.value = None
            self.funcArgs.append(fa)

        for k, v in sig.parameters.items():

            fa = FunctionArg()
            fa.name = v.name
            fa.value = values.parse_instance(None, v.name, v.default)
            self.funcArgs.append(fa)

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'Value', args = [], line = -1):
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

        code = utils.clip_head(inspect.getsource(func))

        self.analyze_args(func)

        self.ast = gast.ast_to_gast(ast.parse(code)).body[0]

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'Value', args = [], line = -1):
        ret = values.UserDefinedInstance(module, None, self.classinfo)
        inst = ret

        func_field = values.Field()
        func_field.set_module(module)

        # add self
        if inst is not None:
            self_func_arg = FunctionArg()
            self_func_arg.value = inst
            args = [self_func_arg] + args

        # add args
        funcArgs = self.parse_args(args)

        for fa in funcArgs:
            func_field.get_field().get_attribute(fa.name).revise(fa.value)

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

        self.analyze_args(func)

        self.ast = gast.ast_to_gast(ast.parse(code)).body[0]

    def vcall(self, module : 'values.Field', graph : 'core.Graph', inst : 'Value', args = [], line = -1):
        func_field = values.Field()
        func_field.set_module(module)

        # add self
        if inst is not None:
            self_func_arg = FunctionArg()
            self_func_arg.value = inst
            args = [self_func_arg] + args

        # add args
        funcArgs = self.parse_args(args)

        for fa in funcArgs:
            func_field.get_field().get_attribute(fa.name).revise(fa.value)

        astc = vevaluator.AstContext(self.ast.body, self.lineno - 1)
        return vevaluator.veval_ast(astc, func_field, graph)
