# coding: utf-8

import chainer
from chainer import cuda
from chainer.initializers import normal
from chainer import variable

# Network definition


class EmbedIDFunction(chainer.function_node.FunctionNode):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def forward(self, inputs):
        x, W = inputs
        self._w_shape = W.shape

        xp = cuda.get_array_module(*inputs)

        if self.ignore_label is not None:
            mask = (x == self.ignore_label)
            return xp.where(mask[..., None], 0, W[xp.where(mask, 0, x)]),

        return W[x],


def embed_id(x, W, ignore_label=None):
    return EmbedIDFunction(ignore_label=ignore_label).apply((x, W))[0]


class EmbedID(chainer.link.Link):
    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None):
        super(EmbedID, self).__init__()
        self.ignore_label = ignore_label

        with self.init_scope():
            if initialW is None:
                initialW = normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))

    def forward(self, x):
        return embed_id(x, self.W, ignore_label=self.ignore_label)


class A(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = EmbedID(n_in, n_out)

    def forward(self, x):
        return self.l1(x)


# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    n_in = 10
    n_out = 20
    model = A(n_in, n_out)

    # print(list(model.namedparams()))
    v = np.random.randint(0, 10, size=5)

    chainer2onnx.generate_testcase(model, [v])
