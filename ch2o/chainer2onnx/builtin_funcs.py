# coding: utf-8

from onnx import helper

from . utils import istensor, new_tensor
from . funcs import Func, totensor


class Builtin_Len(object):
    def call(self, args, _, env):
        assert len(args) == 1
        x = args[0]
        res = new_tensor()
        v = new_tensor()
        env.addnode(
            "Shape",
            inputs=[x.name], outputs=[v.name]
        )

        env.addnode(
            "Gather",
            inputs=[v.name, totensor(0, env).name], outputs=[res.name],
            axis=0
        )

        return res


def builtin_range(args, _, env):
    assert len(args) <= 1  # TODO(satos) 一般的にする

    if not any(map(istensor, args)):
        # print('constant loop',args)
        return range(*args)

    a = new_tensor()
    b = new_tensor()
    res = new_tensor()
    env.addnode(
        'Loop',
        inputs=[args[0].name, ""], outputs=[res.name],
        body=helper.make_graph(
            [],
            "Range_subgraph",
            [a, b], [b, a]
        )
    )
    return res


builtin_functions = {
    'len': Builtin_Len(),
    'range': Func(builtin_range),
}
