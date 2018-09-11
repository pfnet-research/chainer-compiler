# coding: utf-8

from onnx import helper
from onnx import TensorProto

from chainer import functions as F
import chainer
import numpy

from . utils import new_tensor, get_dims, size2d, istensor, totensor

import code


class Function_Relu(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(get_dims(v))
        env.nodes.append(
            helper.make_node(
                "Relu", inputs=[v.name], outputs=[res.name]
            )
        )
        return res


class Function_Pool2d_Util(object):
    def __init__(self, pooltype):
        self.pooltype = pooltype

    def call(self, args, keywords, env):
        assert(len(args) == 2)
        v = args[0]
        res = new_tensor(['TODO'])
        ksize = args[1]
        strides = size2d(keywords.get('stride', ksize))
        # chainer のsize参考
        # https://github.com/chainer/chainer/blob/v4.3.1/chainer/utils/conv.py#L7

        # paddingについて、Chainerの cover_all=Falseと
        # onnx の pads=0 が一致する
        # ので、 cover_all=True(デフォルト)なら
        # padsを入れる必要あり
        pads = [0, 0, 0, 0]
        if keywords.get('cover_all', True):
            dx, dy = 0, 0
            if 'pad' in keywords.keys():
                dy, dx = size2d(keywords['pad'])
            # pads = [dx, dy, strides[0]+dx-1, strides[1]+dy-1]
            # 多めに足しておくとうまいこといくはず (この時点で入力大きさが不明なので)
            # mxnetだと足しておくとよかったが、
            # Onikuだとそうではないっぽい？

            # (size + pad) % stride = ksize % stride
            # を仮定してよい？

            pads = [dx, dy, dx, dy]
        else:
            raise Exception("unimplemented cover_all=False in maxpool2d")

        env.nodes.append(
            helper.make_node(
                self.pooltype, inputs=[v.name], outputs=[res.name],
                kernel_shape=size2d(ksize),
                strides=strides,
                pads=pads
            )
        )
        return res


class Function_MaxPool2d(object):
    def __init__(self):
        self.call = Function_Pool2d_Util('MaxPool').call


class Function_AveragePool2d(object):
    def __init__(self):
        self.call = Function_Pool2d_Util('AveragePool').call


class Function_LocalRespNorm(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['TODO'])
        n = keywords.get('n', 5)
        alpha = keywords.get('alpha', 0.0001)
        env.nodes.append(
            helper.make_node(
                "LRN", inputs=[v.name], outputs=[res.name],
                size=n,
                bias=keywords.get('k', 2.0),
                alpha=alpha * n,  # chainerとonnx(mxnet)で一致しない
                beta=keywords.get('beta', 0.75)
            )
        )
        return res


class Function_Dropout(object):
    def call(self, args, keywords, env):  # たぶん実際には実装できない
        if len(args) == 1:
            pass
        elif len(args) == 2:
            keywords['ratio'] = args[1]
        else:
            raise Exception("invalid length")

        v = args[0]
        res = new_tensor(['TODO'])
        env.nodes.append(
            helper.make_node(
                "Dropout", inputs=[v.name], outputs=[res.name],
                ratio=keywords.get('ratio', 0.5),
            )
        )
        return res


class Function_Concat(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['TODO'])
        # print(list(v))
        env.nodes.append(
            helper.make_node(
                "Concat",
                inputs=list(map(lambda x: x.name, v)), outputs=[res.name],
                axis=keywords.get('axis', 1),
            )
        )
        return res


class Function_SoftmaxCrossEntropy(object):
    def call(self, args, keywords, env):
        assert(len(args) == 2)

        v, w = args[0], args[1]
        # TODO(hamaji): Better to get the dtype of the Tensor from its
        # actual value.
        w.type.tensor_type.elem_type = TensorProto.INT64

        res = new_tensor(get_dims(v))
        env.nodes.append(
            helper.make_node(
                "OnikuxSoftmaxCrossEntropy",
                inputs=[v.name, w.name], outputs=[res.name]
            )
        )
        return res


class Function_PadSequence(object):
    def call(self, args, keywords, env):
        # assert(len(args) == 1)

        v = args[0]

        res = new_tensor()
        env.addnode(
            "OnikuxSequencePad",
            inputs=[v.name], outputs=[res.name]
        )
        return res


class Function_SwapAxes(object):
    def call(self, args, keywords, env):
        assert(len(args) == 3)

        v = args[0]

        a, b = args[1], args[2]
        pe = list(range(max(a, b)+1))
        pe[a] = b
        pe[b] = a

        res = new_tensor()
        env.addnode(
            "Transpose",
            inputs=[v.name], outputs=[res.name],
            perm=pe
        )
        return res


class Function_Reshape(object):
    def call(self, args, keywords, env):
        assert(len(args) == 2)

        v = args[0]
        w = totensor(args[1], env)

        res = new_tensor()
        env.addnode(
            "Reshape",
            inputs=[v.name, w.name], outputs=[res.name],
            perm=[]
        )
        return res


class Function_Tanh(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor()
        env.addnode(
            "Tanh",
            inputs=[v.name], outputs=[res.name],
            perm=[]
        )
        return res


class Np_Array(object):
    def call(self, args, keywords, env):
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


class Np_Int32(object):
    def call(self, args, keywords, env):
        assert len(args) == 1
        v = args[0]
        tt = TensorProto.INT32
        res = new_tensor()
        res.type.tensor_type.elem_type = tt
        env.addnode(
            'Cast',
            inputs=[v.name], outputs=[res.name],
            to=tt
        )
        return res


class Xp_Np_Ceil(object):
    def call(self, args, _, env):
        assert len(args) == 1
        res = new_tensor()
        env.addnode(
            'Ceil',
            inputs=[args[0].name], outputs=[res.name]
        )
        return res


class Cuda_ToCpu(object):
    def call(self, args, _, env):
        assert len(args) == 1
        # TODO(satos) gpuからcpuに移ったというデータをどうにかして載せる
        return args[0]


class Function_Vstack(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        print(args)
        print(list(map(lambda x: x.name, args)))
        return Function_Concat().call([args], {'axis': 0}, env)


class Function_Hstack(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        return Function_Concat().call([args], {'axis': 1}, env)


class Function_Dummy(object):
    def __init__(self, s=""):
        self.name = s

    def call(self, args, keywords, env):
        # raise Exception(self,"Unimplemented")
        env.addnode(
            "Dummy of %s and should be removed" % self.name,
            inputs=[], outputs=[]
        )
        return new_tensor()


dummies = [
    F.vstack,
    F.split_axis,
    F.separate,
    F.stack,
    F.flatten,
    F.accuracy,
    F.squeeze,
    F.broadcast_to,
    F.expand_dims,
    F.softmax,
    F.sum,
    F.hstack,
    numpy.cumsum,
]


class Func(object):
    def __init__(self, f):
        self.call = f


Func2NodeClass = [
    (F.relu, Function_Relu()),
    (F.max_pooling_2d, Function_MaxPool2d()),
    (F.local_response_normalization, Function_LocalRespNorm()),
    (F.dropout, Function_Dropout()),
    (F.concat, Function_Concat()),
    (F.average_pooling_2d, Function_AveragePool2d()),
    (F.softmax_cross_entropy, Function_SoftmaxCrossEntropy()),
    (F.pad_sequence, Function_PadSequence()),
    (F.swapaxes, Function_SwapAxes()),
    (F.reshape, Function_Reshape()),
    (F.tanh, Function_Tanh()),
    (numpy.array, Np_Array()),
    (numpy.ceil, Xp_Np_Ceil()),
    (chainer.backends.cuda.to_cpu, Cuda_ToCpu()),
    (F.vstack, Function_Vstack()),
    (numpy.int32, Np_Int32()),

    # TODO(satos) とりあえずhttps://github.com/espnet/espnet/blob/master/src/nets/deterministic_embe    d_id.py#L43) のif文を通らないようにする
    (chainer.utils.type_check.same_types, Func(lambda _, __, ___: True)),
] + (
    list(map(lambda f: (f, Function_Dummy(f)), dummies))
)
