# coding: utf-8

import onnx
from onnx import helper
from onnx import TensorProto
import os


def new_tensor(dims=['Undefined']):
    tn = new_tensor.cnt
    new_tensor.cnt += 1
    return helper.make_tensor_value_info(
        'T' + str(tn), TensorProto.FLOAT, dims)


new_tensor.cnt = 0


def get_dims(tensor):
    dims = tensor.type.tensor_type.shape.dim
    return list(map(lambda x: x.dim_value, dims))


def clip_head(s):
    s = s.split('\n')
    # print(s)
    hs = os.path.commonprefix(list(filter(lambda x: x != '', s)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    s = map(lambda x: x[ls:], s)
    return '\n'.join(s)


class ValueReturn(Exception):
    def __init__(self, value):
        self.value = value


def size2d(v):
    if isinstance(v, tuple):
        return list(v)
    elif isinstance(v, int):
        return [v, v]
    else:
        raise Exception('size should be tuple or int, but got ', v)


def istensor(x):
    return isinstance(x, onnx.onnx_ONNX_NAMESPACE_ml_pb2.ValueInfoProto)


def totensor(x, env):
    if istensor(x):
        return x
    res = new_tensor()

    assert type(x) == float or type(x) == int
    env.addnode(
        'Constant',
        inputs=[], outputs=[res.name],
        value=onnx.helper.make_tensor(
            name="hoge",
            data_type=(onnx.TensorProto.FLOAT if type(x) ==
                       float else onnx.TensorProto.INT64),
            dims=[],
            vals=[x],
        )
    )
    return res
