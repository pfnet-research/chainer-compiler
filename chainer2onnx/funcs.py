# coding: utf-8

from onnx import helper
from onnx import TensorProto

import chainer
from chainer import functions as F
import numpy

from . utils import new_tensor, get_dims, size2d, istensor, totensor, Env, clip_head

import ast
import code
import gast


class Function_SimpleUnary(object):
    def __init__(self, onnx_name):
        self.onnx_name = onnx_name

    def call(self, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = env.calc(
            self.onnx_name, inputs=[v.name],
        )
        return res


class Function_Pool2d_Util(object):
    def __init__(self, pooltype, hascoverall):
        self.pooltype = pooltype
        self.hascoverall = hascoverall

    def call(self, args, keywords, env):
        assert(len(args) == 2)
        v = args[0]
        res = new_tensor()
        ksize = args[1]
        strides = size2d(keywords.get('stride', ksize))
        # chainer のsize参考
        # https://github.com/chainer/chainer/blob/v4.3.1/chainer/utils/conv.py#L7

        dx, dy = 0, 0
        if 'pad' in keywords.keys():
            dy, dx = size2d(keywords['pad'])

        pads = [dx, dy, dx, dy]

        if self.hascoverall:
            res = env.calc(
                self.pooltype, inputs=[v.name],
                kernel_shape=size2d(ksize),
                strides=strides,
                pads=pads,
                onikux_cover_all=keywords.get('cover_all', True)
            )
        else:
            res = env.calc(
                self.pooltype, inputs=[v.name],
                kernel_shape=size2d(ksize),
                strides=strides,
                pads=pads
            )
        return res


class Function_MaxPool2d(object):
    def __init__(self):
        self.call = Function_Pool2d_Util('MaxPool', True).call


class Function_AveragePool2d(object):
    def __init__(self):
        self.call = Function_Pool2d_Util('AveragePool', False).call


class Function_LocalRespNorm(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        n = keywords.get('n', 5)
        alpha = keywords.get('alpha', 0.0001)
        res = env.calc(
            "LRN", inputs=[v.name],
            size=n,
            bias=keywords.get('k', 2.0),
            alpha=alpha * n,  # chainerとonnx(mxnet)で一致しない
            beta=keywords.get('beta', 0.75)
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
        res = env.calc(
            "Dropout", inputs=[v.name],
            ratio=keywords.get('ratio', 0.5),
        )
        return res


class Function_Concat(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        assert isinstance(v, tuple)  # 今のところ tuple 以外は concat できない
        res = env.calc(
            "Concat",
            inputs=list(map(lambda x: x.name, v)),
            axis=keywords.get('axis', 1),
        )
        return res


class Function_SoftmaxCrossEntropy(object):
    def call(self, args, keywords, env):
        assert(len(args) == 2)

        v, w = args[0], args[1]
        # TODO(hamaji): Better to get the dtype of the Tensor from its
        # actual value.
        w.type.tensor_type.elem_type = TensorProto.INT64

        res = env.calc(
            "OnikuxSoftmaxCrossEntropy",
            inputs=[v.name, w.name],
        )
        return res


class Function_PadSequence(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)

        v = args[0]
        res = env.calc(
            "OnikuxSequencePad",
            inputs=[v.name],
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

        res = env.calc(
            "Transpose",
            inputs=[v.name],
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
        )
        return res


class Function_ExpandDims(object):
    def call(self, args, keywords, env):
        # TODO(hamaji): `axis` may be passed as a positional parameter.
        assert(len(args) == 1)  # axis はキーワード引数でないといけない
        assert 'axis' in keywords.keys()

        v = args[0]

        res = env.calc(
            "Unsqueeze",
            inputs=[v.name],
            axes=[keywords['axis']],
        )
        return res


class Function_BroadcastTo(object):
    def call(self, args, keywords, env):
        assert(len(args) == 2)

        v = args[0]
        w = totensor(args[1], env)

        res = env.calc(
            "Expand",
            inputs=[v.name, w.name]
        )
        return res


def castto(v, tt, env):
    res = new_tensor()
    res.type.tensor_type.elem_type = tt
    env.addnode(
        'Cast',
        inputs=[v.name], outputs=[res.name],
        to=tt
    )
    return res


class Np_Array(object):
    def call(self, args, keywords, env):
        assert len(args) <= 2
        assert 'dtype' in keywords.keys()
        v = args[0]

        t = keywords['dtype']
        if t == numpy.int32:
            tt = TensorProto.INT32
        elif t == numpy.float32:
            tt = TensorProto.FLOAT
        else:
            raise Exception("Unimplemented")

        if istensor(v):
            return castto(v, tt, env)
        else:
            import onnx
            res = env.calc(
                'Constant',
                inputs=[],
                value=onnx.helper.make_tensor(
                    name="hoge",
                    data_type=tt,
                    dims=[],
                    vals=v,
                )
            )
            return res


class Np_Int32(object):
    def call(self, args, keywords, env):
        assert len(args) == 1
        v = args[0]
        return castto(v, TensorProto.INT32, env)


class Np_Cumsum(object):
    def call(self, args, keywords, env):
        assert len(args) == 1
        assert 'axis' not in keywords.keys()
        assert 'dtype' not in keywords.keys()
        # さらにさらに、入力は1次元のTensorである、と仮定してしまいます
        # 戻り値は入力に依らずテンソルらしい
        # TODO(satos) さすがに仮定がきつい
        v = args[0]

        # これ戻り値がテンソルでなくSequenceなら、SplitAxisみたいにかっこよく書けるはず
        """
        a = new_tensor()
        env.addnode(
            'Flatten',
            inputs=[v.name],outputs[a.name],
            axis=0
        )
        v = a
        a = new_tensor()
        env.addnode(
            'Squeeze',
            inputs=[v.name],outputs[a.name],
            axes=[0]
        )
        """
        ls = new_tensor()
        env.addnode(
            'OnikuxGenericLen',
            inputs=[v.name], outputs=[ls.name],
        )

        def dummy():
            return "dummy_" + new_tensor().name

        localenv = Env()
        cnt = new_tensor()
        cond = new_tensor()
        s = new_tensor()
        ts = new_tensor()
        gtx = new_tensor()
        tx = new_tensor()
        localenv.addnode(
            "OnikuxGenericGetItem",
            inputs=[gtx.name, cnt.name], outputs=[tx.name],
        )
        localenv.addnode(
            "Add",
            inputs=[tx.name, s.name], outputs=[ts.name],
        )

        zero = totensor(0, env)

        res = new_tensor()
        env.addnode(
            'Loop',
            inputs=[ls.name, "", v.name, zero.name],
            outputs=[dummy(), dummy(), res.name],
            body=helper.make_graph(
                localenv.nodes,
                "Loop_subgraph",
                [cnt, cond, gtx, s],
                [cond, gtx, ts, ts]
            )
        )

        return res


class Function_SplitAxis(object):
    def call(self, args, keywords, env):
        assert len(args) == 2
        assert keywords['axis'] == 0
        # さらにさらに、入力は1次元のTensorである、と仮定してしまいます
        # 戻り値はtuple(!!)らしいが、たってきSequenceで返してます。
        # TODO(satos) さすがに仮定がきつい

        v = args[0]
        ilens = args[1]

        from . chainer2onnx import eval_ast

        src = """
        r = []
        bs = 0
        for s in ilens:
            r.append(v[bs:s])
            bs = s
        r.append(v[bs:])
        """
        src = clip_head(src)
        nast = gast.ast_to_gast(ast.parse(src))

        localenv = Env()
        localenv.module = {}
        vs = {
            'v': v,
            'ilens': ilens,
        }
        localenv.vars.update(vs)
        eval_ast(nast.body, localenv)

        env.nodes += localenv.nodes
        return localenv.vars['r']


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


# F.vstack, F.hstack は形だけ書いてますがまだ実装していません...
class Function_Vstack(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        raise Exception("This implementation of Vstack seems wrong")
        print(args)
        print(list(map(lambda x: x.name, args)))
        return Function_Concat().call([args], {'axis': 0}, env)


class Function_Hstack(object):
    def call(self, args, keywords, env):
        assert(len(args) == 1)
        raise Exception("This implementation of Hstack is wrong")
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
    F.separate,
    F.stack,
    F.flatten,
    F.accuracy,
    F.squeeze,
    F.softmax,
    F.sum,
    F.hstack,
]


class Func(object):
    def __init__(self, f):
        self.call = f


Func2NodeClass = dict([
    (F.relu, Function_SimpleUnary('Relu')),
    (F.sigmoid, Function_SimpleUnary('Sigmoid')),
    (F.tanh, Function_SimpleUnary('Tanh')),
    (F.max_pooling_2d, Function_MaxPool2d()),
    (F.local_response_normalization, Function_LocalRespNorm()),
    (F.dropout, Function_Dropout()),
    (F.concat, Function_Concat()),
    (F.average_pooling_2d, Function_AveragePool2d()),
    (F.softmax_cross_entropy, Function_SoftmaxCrossEntropy()),
    (F.pad_sequence, Function_PadSequence()),
    (F.swapaxes, Function_SwapAxes()),
    (F.reshape, Function_Reshape()),
    (F.broadcast_to, Function_BroadcastTo()),
    (F.expand_dims, Function_ExpandDims()),
    (numpy.array, Np_Array()),
    (numpy.ceil, Xp_Np_Ceil()),
    (chainer.backends.cuda.to_cpu, Cuda_ToCpu()),
    (F.vstack, Function_Vstack()),
    (numpy.int32, Np_Int32()),
    (numpy.cumsum, Np_Cumsum()),
    (F.split_axis, Function_SplitAxis()),

    # TODO(satos) とりあえずhttps://github.com/espnet/espnet/blob/master/src/nets/deterministic_embe    d_id.py#L43) のif文を通らないようにする
    (chainer.utils.type_check.same_types, Func(lambda _, __, ___: True)),
] + (
    list(map(lambda f: (f, Function_Dummy(f)), dummies))
))
