# coding: utf-8

from onnx import helper
from onnx import TensorProto

from chainer import links as L

from ch2o.callable import Callable
from ch2o.utils import new_tensor, size2d, totensor, Env, clip_head
from ch2o.value import Value

import ast
import code
import gast


class Link_Linear(Callable):
    def __init__(self, ch):
        super(Link_Linear, self).__init__(L.Linear(None))

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
            '/W', TensorProto.FLOAT,
            [self.n_out, ('input_size' if (self.n_in is None) else self.n_in)])

        if not self.nobias:
            self.b = helper.make_tensor_value_info(
                '/b', TensorProto.FLOAT, [self.n_out])

    def call_impl(self, env, x):
        x = x.to_tensor(env)
        res = new_tensor([self.n_out])

        if self.nobias:
            t = env.calc(
                "Transpose",
                inputs=[self.W.name],
                perm=[1, 0]
            )
            res = env.calc(
                "MatMul",
                inputs=[x.name, t.name],
            )
        else:
            res = env.calc(
                "Gemm",
                inputs=[x.name, self.W.name, self.b.name],
                transA=0, transB=1
            )
        return res

    def init_tensors(self):
        if self.nobias:
            return [self.W]
        else:
            return [self.W, self.b]


class Link_Convolution2D(Callable):
    def __init__(self, ch):
        super(Link_Convolution2D, self).__init__(L.Convolution2D(None, None))

        # code.InteractiveConsole({'ch': ch}).interact()

        self.ksize = size2d(ch.ksize)
        self.stride = size2d(ch.stride)
        ps = size2d(ch.pad)
        self.pads = ps + ps

        if not (ch.b is None):
            # nobias = True の場合
            self.M = ch.b.shape[0]
            self.b = helper.make_tensor_value_info(
                '/b', TensorProto.FLOAT, [self.M])
        else:
            self.M = "TODO"
            self.b = None

        self.W = helper.make_tensor_value_info(
            '/W', TensorProto.FLOAT,
            [self.M, 'channel_size'] + list(self.ksize))

    def call_impl(self, env, x):
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Conv",
                inputs=[x.to_tensor(env).name, self.W.name] +
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


class Link_BatchNormalization(Callable):
    def __init__(self, ch):
        super(Link_BatchNormalization, self).__init__(
            L.BatchNormalization(1))

        self.n_out = ch.beta.shape[0]

        self.scale = helper.make_tensor_value_info(
            '/gamma', TensorProto.FLOAT, [self.n_out])
        self.B = helper.make_tensor_value_info(
            '/beta', TensorProto.FLOAT, [self.n_out])
        self.mean = helper.make_tensor_value_info(
            '/avg_mean', TensorProto.FLOAT, [self.n_out])
        self.var = helper.make_tensor_value_info(
            '/avg_var', TensorProto.FLOAT, [self.n_out])

        self.eps = ch.eps
        self.momentum = ch.decay

    def call_impl(self, env, x, **kwargs):
        assert not kwargs  # TODO(hamaji): finetune not supported yet.
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "BatchNormalization",
                inputs=[x.to_tensor(env).name, self.scale.name, self.B.name,
                        self.mean.name, self.var.name], outputs=[res.name],
                epsilon=self.eps,
                momentum=self.momentum,
                # とりあえずspatialは1で(0でも値が変わらなかったのでよくわからん)
            )
        )
        return res

    def init_tensors(self):
        return [self.scale, self.B, self.mean, self.var]


class Link_NStepLSTM(Callable):
    def __init__(self, ch):
        super(Link_NStepLSTM, self).__init__(L.NStepLSTM(1, 1, 1, 0))

        hd = ch.children().__next__()
        if not(hd.w0 is None):
            self.n_in = hd.w0.shape[1]
        else:
            self.n_in = None

        self.out_size = ch.out_size
        self.n_layers = ch.n_layers
        self.dropout = ch.dropout

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            ws = []
            bs = []
            for j in range(8):
                ws.append(helper.make_tensor_value_info(
                    ('/%d/w%d' % (i, j)), TensorProto.FLOAT, ["TODO"]))
                bs.append(helper.make_tensor_value_info(
                    ('/%d/b%d' % (i, j)), TensorProto.FLOAT, ["TODO"]))
            self.ws.append(ws)
            self.bs.append(bs)

    def call_impl(self, env, hx, cx, xs):
        assert hx.value is None  # TODO(hamaji): Not implemented yet.
        assert cx.value is None  # TODO(hamaji): Not implemented yet.
        xs = xs.to_sequence(env)

        # とりあえずnstep を 1step ずつに分解する
        ilens = env.calc(
            "OnikuxSequenceLengths",
            inputs=[xs.name],
        )

        tilens = env.calc(
            "OnikuxSequenceStack",
            inputs=[ilens.name],
        )

        v = env.calc(
            "OnikuxSequencePad",
            inputs=[xs.name],
        )
        v = env.calc(
            "Transpose",
            perm=(1, 0, 2),
            inputs=[v.name],
        )

        def lstm_param(ps):
            p = env.calc(
                "Concat",
                inputs=[v.name for v in ps],
                axis=0
            )
            return env.calc(
                "Unsqueeze",
                inputs=[p.name],
                axes=[0]
            )

        ws = []
        rs = []
        bs = []
        for w in self.ws:
            ws.append(lstm_param([w[0], w[3], w[1], w[2]]))
            rs.append(lstm_param([w[4], w[7], w[5], w[6]]))
        for b in self.bs:
            bs.append(lstm_param([b[0], b[3], b[1], b[2],
                                  b[4], b[7], b[5], b[6]]))

        hs = []
        cs = []
        for i in range(self.n_layers):
            h = new_tensor()
            c = new_tensor()
            ys = new_tensor()

            env.addnode(
                "LSTM",
                inputs=[v.name, ws[i].name, rs[i].name, bs[i].name,
                        tilens.name],
                outputs=[ys.name, h.name, c.name],
                direction='forward',
                hidden_size=self.out_size,
                # sequence_lens=[ilens.name]
            )

            hs.append(h.name)
            cs.append(c.name)
            yys = env.calc(
                "Squeeze",
                inputs=[ys.name],
                axes=[1]
            )
            v = yys
        # print(hs)
        # print(cs)
        ths = env.calc(
            "Concat",
            inputs=hs,
            axis=0,
        )
        tcs = env.calc(
            "Concat",
            inputs=cs,
            axis=0,
        )

        tv = env.calc(
            "Transpose",
            perm=(1, 0, 2),
            inputs=[v.name],
        )
        v = tv

        tys = env.calc(
            "OnikuxSequenceUnpad",
            inputs=[v.name, ilens.name],
        )
        return ths, tcs, tys

    def init_tensors(self):
        tensors = []
        for w in self.ws:
            tensors += w
        for b in self.bs:
            tensors += b
        return tensors


class Link_NStepBiLSTM(Callable):
    def __init__(self, ch):
        super(Link_NStepBiLSTM, self).__init__(L.NStepBiLSTM(1, 1, 1, 0))
        # code.InteractiveConsole({'ch': ch}).interact()

        hd = ch.children().__next__()
        if not(hd.w0 is None):
            self.n_in = hd.w0.shape[1]
        else:
            self.n_in = None

        self.out_size = ch.out_size
        self.n_layers = ch.n_layers
        self.dropout = ch.dropout

        self.ws = []
        self.bs = []
        for i in range(self.n_layers * 2):
            ws = []
            bs = []
            for j in range(8):
                ws.append(helper.make_tensor_value_info(
                    ('/%d/w%d' % (i, j)), TensorProto.FLOAT, ["TODO"]))
                bs.append(helper.make_tensor_value_info(
                    ('/%d/b%d' % (i, j)), TensorProto.FLOAT, ["TODO"]))
            self.ws.append(ws)
            self.bs.append(bs)

    def call_impl(self, env, hx, cx, xs):
        assert hx.value is None  # TODO(hamaji): Not implemented yet.
        assert cx.value is None  # TODO(hamaji): Not implemented yet.
        xs = xs.to_sequence(env)

        # とりあえずnstep を 1step ずつに分解する
        ilens = env.calc(
            "OnikuxSequenceLengths",
            inputs=[xs.name],
        )

        tilens = env.calc(
            "OnikuxSequenceStack",
            inputs=[ilens.name],
        )

        v = xs

        def lstm_param(ps):
            p = env.calc(
                "Concat",
                inputs=[v.name for v in ps],
                axis=0
            )
            return env.calc(
                "Unsqueeze",
                inputs=[p.name],
                axes=[0]
            )

        wst = []
        rst = []
        bst = []
        for w in self.ws:
            wst.append(lstm_param([w[0], w[3], w[1], w[2]]))
            rst.append(lstm_param([w[4], w[7], w[5], w[6]]))
        for b in self.bs:
            bst.append(lstm_param([b[0], b[3], b[1], b[2],
                                   b[4], b[7], b[5], b[6]]))

        ws = []
        rs = []
        bs = []
        for i in range(self.n_layers):
            for s, t in [(ws, wst), (rs, rst), (bs, bst)]:
                s.append(env.calc(
                    "Concat",
                    inputs=[t[i*2].name, t[i*2+1].name],
                    axis=0
                ))

        hs = []
        cs = []
        for i in range(self.n_layers):
            v = Value(v).to_sequence(env)
            v = env.calc(
                "OnikuxSequencePad",
                inputs=[v.name],
            )
            v = env.calc(
                "Transpose",
                perm=(1, 0, 2),
                inputs=[v.name]
            )

            h = new_tensor()
            c = new_tensor()
            ys = new_tensor()

            env.addnode(
                "LSTM",
                inputs=[v.name, ws[i].name,
                        rs[i].name, bs[i].name, tilens.name],
                outputs=[ys.name, h.name, c.name],
                direction='bidirectional',
                hidden_size=self.out_size,
            )

            hs.append(h.name)
            cs.append(c.name)

            # ys :: seqlen * 2 * batchsize * hiddensize
            v = env.calc("Transpose", perm=(2, 0, 1, 3), inputs=[ys.name])
            v = env.calc_seq("OnikuxSequenceUnpad", inputs=[v.name, ilens.name])

            from . chainer2onnx import eval_ast
            import chainer
            localenv = Env({})
            vs = {
                'v': Value(v),
                'F': chainer.functions
            }
            localenv.update_vars(vs)
            src = """
            r = []
            for d in v:
                r.append(F.reshape(d,(-1,%d)))
            v = r
            """ % (2 * self.out_size)
            src = clip_head(src)
            nast = gast.ast_to_gast(ast.parse(src))
            eval_ast(nast.body, localenv)

            env.nodes += localenv.nodes
            v = localenv.get_var('v')

        ths = env.calc(
            "Concat",
            inputs=hs,
            axis=1,
        )
        tcs = env.calc(
            "Concat",
            inputs=cs,
            axis=1,
        )

        ths = env.calc("Transpose", inputs=[ths.name], perm=(1, 0, 2))
        tcs = env.calc("Transpose", inputs=[tcs.name], perm=(1, 0, 2))

        tys = v
        return ths, tcs, tys

    def init_tensors(self):
        tensors = []
        for w in self.ws:
            tensors += w
        for b in self.bs:
            tensors += b
        return tensors


class Link_EmbedID(Callable):
    def __init__(self, ch):
        super(Link_EmbedID, self).__init__(L.EmbedID(1, 1))

        self.n_vocab = ch.W.shape[0]
        self.n_out = ch.W.shape[1]

        self.W = helper.make_tensor_value_info(
            '/W', TensorProto.FLOAT, list(ch.W.shape))

    def call_impl(self, env, x):
        res = env.calc(
            "Gather",
            inputs=[self.W.name, x.to_tensor(env).name],
        )
        return res

    def init_tensors(self):
        return [self.W]


class Link_StatelessLSTM(Callable):
    def __init__(self, ch, parentname):
        super(Link_StatelessLSTM, self).__init__(L.StatelessLSTM(1))

        self.name = ''
        # code.InteractiveConsole({'ch': ch}).interact()

        self.upward = Link_Linear(ch.upward, self.name)
        self.lateral = Link_Linear(ch.lateral, self.name)

    def call_impl(self, env, c, h, x):
        # TODO(satos) 正しくする(ただただ面倒だが)
        # とりあえずnstep を 1step ずつに分解する
        # print(self.name,args)
        # assert(len(args) == 1)

        return new_tensor(), new_tensor()

    def init_tensors(self):
        return self.upward.init_tensors() + self.lateral.init_tensors()


Link2NodeClass = {
    L.Linear: Link_Linear,
    L.Convolution2D: Link_Convolution2D,
    L.BatchNormalization: Link_BatchNormalization,
    L.NStepLSTM: Link_NStepLSTM,
    L.NStepBiLSTM: Link_NStepBiLSTM,
    L.EmbedID: Link_EmbedID,
    L.StatelessLSTM: Link_StatelessLSTM,
}
