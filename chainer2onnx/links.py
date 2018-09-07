# coding: utf-8

from onnx import helper
from onnx import TensorProto

from chainer import links as L

from . utils import new_tensor, size2d

import code


class Link_Linear(object):
    def __init__(self, ch, parentname):
        # code.InteractiveConsole({'ch': ch}).interact()
        self.name = parentname + '_' + ch.name

        if ch.b is None:
            self.n_out = 'output_size'
            self.nobias = True
        else:
            self.n_out = ch.b.shape[0]
            self.nobias = False

        if not(ch.W.data is None):
            self.n_in = ch.W.shape[1]
        else:
            self.n_in = None

        self.W = helper.make_tensor_value_info(
            self.name + '_W', TensorProto.FLOAT,
            [self.n_out, ('input_size' if (self.n_in is None) else self.n_in)])

        if not self.nobias:
            self.b = helper.make_tensor_value_info(
                self.name + '_b', TensorProto.FLOAT, [self.n_out])

    def call(self, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor([self.n_out])

        if self.nobias:
            x = new_tensor()
            env.addnode(
                "Transpose",
                inputs=[self.W.name], outputs=[x.name],
                perm=[1, 0]
            )
            env.addnode(
                "MatMul",
                inputs=[v.name, x.name], outputs=[res.name]
            )
        else:
            env.addnode(
                "Gemm",
                inputs=[v.name, self.W.name, self.b.name], outputs=[res.name],
                transA=0, transB=1
            )
        return res

    def init_tensors(self):
        if self.nobias:
            return [self.W]
        else:
            return [self.W, self.b]


class Link_Convolution2D(object):
    def __init__(self, ch, parentname):
        self.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        self.ksize = size2d(ch.ksize)
        self.stride = size2d(ch.stride)
        ps = size2d(ch.pad)
        self.pads = ps + ps

        if not (ch.b is None):
            # nobias = True の場合
            self.M = ch.b.shape[0]
            self.b = helper.make_tensor_value_info(
                self.name + '_b', TensorProto.FLOAT, [self.M])
        else:
            self.M = "TODO"
            self.b = None

        self.W = helper.make_tensor_value_info(
            self.name + '_W', TensorProto.FLOAT,
            [self.M, 'channel_size'] + self.ksize)

    def call(self, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Conv",
                inputs=[v.name, self.W.name] +
                ([] if self.b is None else [self.b.name]),
                outputs=[res.name],
                kernel_shape=self.ksize,
                pads=self.pads,
                strides=self.stride
            )
        )
        return res

    def init_tensors(self):
        return [self.W] + ([] if self.b is None else [self.b])


class Link_BatchNormalization(object):
    def __init__(self, ch, parentname):
        self.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        self.n_out = ch.beta.shape[0]

        self.scale = helper.make_tensor_value_info(
            self.name + '_gamma', TensorProto.FLOAT, [self.n_out])
        self.B = helper.make_tensor_value_info(
            self.name + '_beta', TensorProto.FLOAT, [self.n_out])
        self.mean = helper.make_tensor_value_info(
            self.name + '_avg_mean', TensorProto.FLOAT, [self.n_out])
        self.var = helper.make_tensor_value_info(
            self.name + '_avg_var', TensorProto.FLOAT, [self.n_out])

        self.eps = ch.eps
        self.momentum = ch.decay

    def call(self, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "BatchNormalization",
                inputs=[v.name, self.scale.name, self.B.name,
                        self.mean.name, self.var.name], outputs=[res.name],
                epsilon=self.eps,
                momentum=self.momentum,
                # とりあえずspatialは1で(0でも値が変わらなかったのでよくわからん)
            )
        )
        return res

    def init_tensors(self):
        return [self.scale, self.B, self.mean, self.var]


class Link_NstepLSTM(object):
    def __init__(self, ch, parentname):
        self.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        # cs = list(ch.children())
        hd = ch.children().__next__()
        if not(hd.w0 is None):
            self.n_in = hd.w0.shape[1]
        else:
            self.n_in = None

        self.out_size = ch.out_size
        self.n_layers = ch.n_layers
        self.dropout = ch.dropout

        class step(object):
            def __init__(self):
                pass

        self.ws = [step() for _ in range(self.n_layers)]
        for i in range(self.n_layers):
            self.ws[i].W = helper.make_tensor_value_info(
                self.name + ('_%d_ws0' % i), TensorProto.FLOAT, ["TODO"])
            # これ多分うまいこと変換しないといけない
            # chainer : at  ct
            #   onnx  : ct  Ct
            # (chainerのws[0],ws[2],ws[1],ws[3]から連結させたりする)
            self.ws[i].R = helper.make_tensor_value_info(
                self.name + ('_%d_ws1' % i), TensorProto.FLOAT, ["TODO"])
            # (chainerのws[4],ws[6],ws[5],ws[7]から連結させたりする)
            self.ws[i].B = helper.make_tensor_value_info(
                self.name + ('_%d_bss' % i), TensorProto.FLOAT, ["TODO"])
            # (chainerのbs[0,2,1,3,4,6,5,7]から連結させたりする)

    def call(self, args, _, env):
        # とりあえずnstep を 1step ずつに分解する
        # print(self.name,args)
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

        for i in range(self.n_layers):

            h = new_tensor(['unknown', 'unknown', 'unknown'])
            c = new_tensor(['unknown', 'unknown', 'unknown'])
            ys = new_tensor(['unknown', 'unknown', 'unknown'])

            env.nodes.append(
                helper.make_node(
                    "LSTM",

                    inputs=[v.name, self.ws[i].W.name,
                            self.ws[i].R.name, self.ws[i].B.name],
                    outputs=[ys.name, h.name, c.name],
                    hidden_size=self.out_size
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

    def init_tensors(self):
        return sum([[self.ws[i].W, self.ws[i].B, self.ws[i].R] for i in range(self.n_layers)], [])


class Link_EmbedID(object):
    def __init__(self, ch, parentname):
        self.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        self.n_vocab = ch.W.shape[0]
        self.n_out = ch.W.shape[1]

        self.W = helper.make_tensor_value_info(
            self.name + '_W', TensorProto.FLOAT, list(ch.W.shape))

    def call(self, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Gather",
                inputs=[self.W.name, v.name], outputs=[res.name],
            )
        )
        return res

    def init_tensors(self):
        return [self.W]


class Link_StatelessLSTM(object):
    def __init__(self, ch, parentname):
        self.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        self.upward = Link_Linear(ch.upward, self.name)
        self.lateral = Link_Linear(ch.lateral, self.name)

    def call(self, args, _, env):
        # TODO(satos) 正しくする(ただただ面倒だが)
        # とりあえずnstep を 1step ずつに分解する
        # print(self.name,args)
        # assert(len(args) == 1)
        
        return new_tensor(),new_tensor()

    def init_tensors(self):
        return self.upward.init_tensors() + self.lateral.init_tensors()


Link2NodeClass = [
    (L.Linear, Link_Linear),
    (L.Convolution2D, Link_Convolution2D),
    (L.BatchNormalization, Link_BatchNormalization),
    (L.NStepLSTM, Link_NstepLSTM),
    (L.EmbedID, Link_EmbedID),
    (L.NStepBiLSTM, Link_NstepLSTM),  # TODO(satos) はりぼてなのでなおす
    (L.StatelessLSTM, Link_StatelessLSTM),
]
