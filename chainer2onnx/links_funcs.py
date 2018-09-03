# coding: utf-8

from onnx import helper
from onnx import TensorProto

from chainer import functions as F
from chainer import links as L

from . utils import new_tensor
from . utils import get_dims


class Link_Linear(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        sl.n_out = ch.b.shape[0]
        if not(ch.W.data is None):
            sl.n_in = ch.W.shape[1]
        else:
            sl.n_in = None

        sl.W = helper.make_tensor_value_info(
            sl.name + '_W', TensorProto.FLOAT,
            [sl.n_out, ('input_size' if (sl.n_in is None) else sl.n_in)])
        sl.b = helper.make_tensor_value_info(
            sl.name + '_b', TensorProto.FLOAT, [sl.n_out])

    def call(sl, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor([sl.n_out])
        env.nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[v.name, sl.W.name, sl.b.name], outputs=[res.name],
                transA=0, transB=1
            )
        )
        return res

    def init_tensors(sl):
        return [sl.W, sl.b]


def size2d(v):
    if isinstance(v, tuple):
        return list(v)
    elif isinstance(v, int):
        return [v, v]
    else:
        raise Exception('size should be tuple or int, but got ', v)


class Link_Convolution2D(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        sl.ksize = size2d(ch.ksize)
        sl.stride = size2d(ch.stride)
        ps = size2d(ch.pad)
        sl.pads = ps + ps

        if not (ch.b is None):
            # nobias = True の場合
            sl.M = ch.b.shape[0]
            sl.b = helper.make_tensor_value_info(
                sl.name + '_b', TensorProto.FLOAT, [sl.M])
        else:
            sl.M = "TODO"
            sl.b = None

        sl.W = helper.make_tensor_value_info(
            sl.name + '_W', TensorProto.FLOAT,
            [sl.M, 'channel_size'] + sl.ksize)

    def call(sl, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Conv",
                inputs=[v.name, sl.W.name] +
                ([] if sl.b is None else [sl.b.name]),
                outputs=[res.name],
                kernel_shape=sl.ksize,
                pads=sl.pads,
                strides=sl.stride
            )
        )
        return res

    def init_tensors(sl):
        return [sl.W] + ([] if sl.b is None else [sl.b])


class Link_BatchNormalization(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        sl.n_out = ch.beta.shape[0]

        sl.scale = helper.make_tensor_value_info(
            sl.name + '_gamma', TensorProto.FLOAT, [sl.n_out])
        sl.B = helper.make_tensor_value_info(
            sl.name + '_beta', TensorProto.FLOAT, [sl.n_out])
        sl.mean = helper.make_tensor_value_info(
            sl.name + '_avg_mean', TensorProto.FLOAT, [sl.n_out])
        sl.var = helper.make_tensor_value_info(
            sl.name + '_avg_var', TensorProto.FLOAT, [sl.n_out])

        sl.eps = ch.eps
        sl.momentum = ch.decay

    def call(sl, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "BatchNormalization",
                inputs=[v.name, sl.scale.name, sl.B.name,
                        sl.mean.name, sl.var.name], outputs=[res.name],
                epsilon=sl.eps,
                momentum=sl.momentum,
                # とりあえずspatialは1で(0でも値が変わらなかったのでよくわからん)
            )
        )
        return res

    def init_tensors(sl):
        return [sl.scale, sl.B, sl.mean, sl.var]


class Link_NstepLSTM(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        cs = list(ch.children())
        sl.n_layers = ch.n_layers
        if not(cs[0].w0 is None):
            sl.n_in = cs[0].w0.shape[1]
        else:
            sl.n_in = None

        sl.out_size = ch.out_size
        sl.dropout = ch.dropout

        class step(object):
            def __init__(sl):
                pass

        sl.ws = [step() for _ in range(sl.n_layers)]
        for i in range(sl.n_layers):
            sl.ws[i].W = helper.make_tensor_value_info(
                sl.name + ('_%d_ws0' % i), TensorProto.FLOAT, ["TODO"])
            # これ多分うまいこと変換しないといけない
            # chainer : at  ct
            #   onnx  : ct  Ct
            # (chainerのws[0],ws[2],ws[1],ws[3]から連結させたりする)
            sl.ws[i].R = helper.make_tensor_value_info(
                sl.name + ('_%d_ws1' % i), TensorProto.FLOAT, ["TODO"])
            # (chainerのws[4],ws[6],ws[5],ws[7]から連結させたりする)
            sl.ws[i].B = helper.make_tensor_value_info(
                sl.name + ('_%d_bss' % i), TensorProto.FLOAT, ["TODO"])
            # (chainerのbs[0,2,1,3,4,6,5,7]から連結させたりする)

    def call(sl, args, _, env):
        # とりあえずnstep を 1step ずつに分解する
        # print(sl.name,args)
        # assert(len(args) == 1)
        assert(args[0] is None and args[1] is None)

        # v = args[2]
        v = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Transpose",
                perm=(1, 0, 2),
                inputs=[args[2].name],
                outputs=[v.name],
            )
        )

        hs = []
        cs = []

        for i in range(sl.n_layers):

            h = new_tensor(['unknown', 'unknown', 'unknown'])
            c = new_tensor(['unknown', 'unknown', 'unknown'])
            ys = new_tensor(['unknown', 'unknown', 'unknown'])

            env.nodes.append(
                helper.make_node(
                    "LSTM",

                    inputs=[v.name, sl.ws[i].W.name,
                            sl.ws[i].R.name, sl.ws[i].B.name],
                    outputs=[ys.name, h.name, c.name],
                    hidden_size=sl.out_size
                )
            )

            hs.append(h.name)
            cs.append(c.name)
            v = ys
        # print(hs)
        # print(cs)
        ths = new_tensor(['unknown', 'unknown', 'unknown'])
        tcs = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Concat",
                inputs=hs, outputs=[ths.name],
                axis=0,
            )
        )
        env.nodes.append(
            helper.make_node(
                "Concat",
                inputs=cs, outputs=[tcs.name],
                axis=0,
            )
        )

        tys = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Transpose",
                perm=(1, 0, 2),
                inputs=[v.name],
                outputs=[tys.name],
            )
        )
        return ths, tcs, tys

    def init_tensors(sl):
        return sum([[sl.ws[i].W, sl.ws[i].B, sl.ws[i].R] for i in range(sl.n_layers)], [])


class Link_EmbedID(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        sl.n_vocab = ch.W.shape[0]
        sl.n_out = ch.W.shape[1]

        sl.W = helper.make_tensor_value_info(
            sl.name + '_W', TensorProto.FLOAT, list(ch.W.shape))

    def call(sl, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Gather",
                inputs=[sl.W.name, v.name], outputs=[res.name],
            )
        )
        return res

    def init_tensors(sl):
        return [sl.W]


class Function_Relu(object):
    def call(sl, args, keywords, env):
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
    def __init__(sl, pooltype):
        sl.pooltype = pooltype

    def call(sl, args, keywords, env):
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
                sl.pooltype, inputs=[v.name], outputs=[res.name],
                kernel_shape=size2d(ksize),
                strides=strides,
                pads=pads
            )
        )
        return res


class Function_MaxPool2d(object):
    def __init__(sl):
        sl.call = Function_Pool2d_Util('MaxPool').call


class Function_AveragePool2d(object):
    def __init__(sl):
        sl.call = Function_Pool2d_Util('AveragePool').call


class Function_LocalRespNorm(object):
    def call(sl, args, keywords, env):
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
    def call(sl, args, keywords, env):  # たぶん実際には実装できない
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
    def call(sl, args, keywords, env):
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
    def call(sl, args, keywords, env):
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


Func2NodeClass = [
    (F.relu, Function_Relu),
    (F.max_pooling_2d, Function_MaxPool2d),
    (F.local_response_normalization, Function_LocalRespNorm),
    (F.dropout, Function_Dropout),
    (F.concat, Function_Concat),
    (F.average_pooling_2d, Function_AveragePool2d),
    (F.softmax_cross_entropy, Function_SoftmaxCrossEntropy)
]

Link2NodeClass = [
    (L.Linear, Link_Linear),
    (L.Convolution2D, Link_Convolution2D),
    (L.BatchNormalization, Link_BatchNormalization),
    (L.NStepLSTM, Link_NstepLSTM),
    (L.EmbedID, Link_EmbedID)
]


class Func(object):
    def __init__(sl, f):
        sl.call = f
