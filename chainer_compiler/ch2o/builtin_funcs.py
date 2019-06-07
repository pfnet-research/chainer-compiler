# coding: utf-8

from onnx import helper

from chainer_compiler.ch2o.callable import Callable
from chainer_compiler.ch2o.funcs import Func, totensor
from chainer_compiler.ch2o.utils import istensor, new_tensor


class Builtin_Len(Callable):
    def __init__(self):
        super(Builtin_Len, self).__init__(lambda x: x)

    def call_impl(self, env, x):
        x = x.to_value_info(env)
        return env.calc(
            "ChainerGenericLen",
            inputs=[x.name],
        )


class Builtin_List(Callable):
    def __init__(self):
        super(Builtin_List, self).__init__(lambda x: x)

    def call_impl(self, env, x):
        return env.calc(
            "Identity",
            inputs=[x.to_sequence(env).name],
        )


def builtin_range(args, _, env):
    if all(a.is_py for a in args):
        # print('constant loop',args)
        return range(*(a.value for a in args))

    return env.calc_seq(
        'ChainerSequenceRange',
        inputs=[a.to_tensor(env).name for a in args]
    )


builtin_functions = {
    len: Builtin_Len(),
    list: Builtin_List(),
    range: Func(builtin_range),
}
