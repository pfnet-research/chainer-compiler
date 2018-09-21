# coding: utf-8

import chainer
import chainer.functions as F
import numpy as np


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs, ilens):
        y1 = F.split_axis(xs, ilens, axis=0)
        # この時点でTuple!! なのでrange based for でlistにする
        y1 = [x for x in y1]
        return y1

# ======================================


import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(12)

    model = A()

    xs = np.random.rand(20).astype(np.float32)
    ilens = [1, 3, 5, 8, 14]

    chainer2onnx.generate_testcase(model, [xs, ilens])
