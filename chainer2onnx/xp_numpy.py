# coding: utf-8

import onnx

import numpy

from . utils import new_tensor, istensor
from . funcs import Func, Function_Dummy


def xp_array(args, _, env):
    assert len(args) <= 2
    # TODO(satos) 型のこと考える
    v = args[0]
    res = new_tensor()
    if istensor(v):
        return v
    env.addnode(
        'Constant',
        inputs=[], outputs=[res.name],
        value=onnx.helper.make_tensor(
            name="hoge",
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=v,
        )
    )
    return res


def xp_ceil(args, _, env):
    assert len(args) == 1
    res = new_tensor()
    env.addnode(
        'Ceil',
        inputs=[args[0].name], outputs=[res.name]
    )
    return res


xp_attrs = {
    'array': Func(xp_array),
    'ceil': Func(xp_ceil),
}

np_attrs = {
    'float32': numpy.float32,
    'int32': numpy.int32,
    'cumsum': Function_Dummy(),
}
