# coding: utf-8

import chainer
import chainer.functions as F

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        # TODO(satos) テストケース増やす
        # y1 = F.max_pooling_2d(x, (1, 3), stride=(1, 4), pad=(0, 1))
        y1 = F.max_pooling_2d(x, (1, 3), stride=(1, 4))
        return y1


# ======================================

import chainer2onnx 
import numpy as np

if __name__ == '__main__':
    
    model = A()
    
    x = v = np.random.rand(2, 3, 1, 13).astype(np.float32)
    chainer2onnx.generate_testcase(model, [x])
