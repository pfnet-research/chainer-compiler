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

    if type(x) == float or type(x) == int:
        if type(x) == float:
            dt = onnx.TensorProto.FLOAT
        else:
            dt = onnx.TensorProto.INT64
        env.addnode(
            'Constant',
            inputs=[], outputs=[res.name],
            value=onnx.helper.make_tensor(
                name="hoge",
                data_type=dt,
                dims=[],
                vals=[x],
            )
        )
    elif type(x) == tuple or type(x) == list:
        def f(v):
            tv = totensor(v, env)
            tw = new_tensor()
            env.addnode(
                'Unsqueeze',
                inputs=[tv.name], outputs=[tw.name],
                axes=[0]
            )
            return tw.name

        vs = list(map(f, x))
        # print(vs)
        env.addnode(
            'Concat',
            inputs=vs, outputs=[res.name],
            axis=0
        )
    else:
        raise Exception("totensor of %s is not implemented yet" % str(x))

    return res


class Env(object):
    def __init__(self):
        self.vars = {}
        self.nodes = []
        self.init_tensors = []
        self.restore_funcs = []  # User定義Linkの初期化子を正常化させるやつ

    def localenv(self):
        res = Env()
        res.nodes = self.nodes  # こっちはglobalに共通でないといけない
        res.init_tensors = self.init_tensors  # こっちも共通
        res.restore_funcs = self.restore_funcs
        return res

    def addnode(self, *args, **kwargs):
        self.nodes.append(
            helper.make_node(*args, **kwargs)
        )

    def add_init(self, inits, pathname):
        for v in inits:
            # drint('add_init',v,p)
            v.name = pathname + v.name
            self.init_tensors.append(v)

    def calc(self, *args, **kwargs):
        res = new_tensor()
        assert 'outputs' not in kwargs.keys()
        kwargs['outputs'] = [res.name]
        self.addnode(*args, **kwargs)
        return res
