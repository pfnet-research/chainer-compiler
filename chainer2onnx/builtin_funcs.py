from . utils import istensor, new_tensor
from . funcs import Func


def builtin_len(args, _, env):
    assert len(args) == 1
    res = new_tensor()
    # TODO(satos) にゃーん
    env.addnode(
        'HogeLen',
        inputs=[args[0].name], outputs=[res.name]
    )
    return res


def builtin_range(args, _, env):
    assert len(args) <= 2  # TODO(satos) 一般的にする

    if not any(map(istensor, args)):
        # print('constant loop',args)
        return range(*args)

    res = new_tensor()
    env.addnode(
        'OnikuxRange',
        inputs=list(map(lambda x: x.name, args)), outputs=[res.name]
    )
    return res


builtin_functions = {
    'len': Func(builtin_len),
    'range': Func(builtin_range),
}
