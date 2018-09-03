# coding: utf-8

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
    def __init__(sl, value):
        sl.value = value
