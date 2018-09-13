# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class F(object):
    def __init__(self,a):
        self.a = a

    def g(self,x):
        return self.a + x

def h(x,y):
    return x + y

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()
 
    def forward(self, x,y,z):
        p = F(x).g(y)
        return h(p,z)


# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    a = np.random.rand(3,4).astype(np.float32)
    b = np.random.rand(3,4).astype(np.float32)
    c = np.random.rand(3,4).astype(np.float32)
    chainer2onnx.generate_testcase(model, [a,b,c])
