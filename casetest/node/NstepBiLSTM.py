# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self, n_layer, n_in, n_out):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepBiLSTM(n_layer, n_in, n_out, 0.1)

    def forward(self, x):
        hy, cs, ys = self.l1(None, None, x)
        return hy, cs, ys
        # return hy,cs


# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    n_batch = 7
    n_layer = 3
    n_in = 8
    n_hidden = 5
    n_time_length = 4
    model = A(n_layer, n_in, n_hidden)

    x = [np.random.rand(n_time_length , n_in).astype(np.float32) for i in range(n_batch)]
    x = [x]
    chainer2onnx.generate_testcase(model, x)
