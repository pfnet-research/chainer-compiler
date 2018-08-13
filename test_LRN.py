# coding: utf-8

import chainer
import chainer.functions as F

# Network definition


class LRN(chainer.Chain):

    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        return F.local_response_normalization(x)


# ======================================from MLP

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    import test_mxnet

    model = LRN()

    v = np.random.rand(1, 11, 1, 1).astype(np.float32)
    test_mxnet.check_compatibility(model, v)
