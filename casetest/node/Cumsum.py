# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F

# Network definition

    

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs, xss, ps):
        y7 = xss[3,ps[0]]
        #print(ps,xs[ps[0]:ps[0]+3])
        # ys = [xs[p] for p in ps]
        return y1,y2,y3,y4,y5,y6,y7

# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(12)

    n_maxlen = 10

    model = A()
    
    u = np.random.rand(n_maxlen+6).astype(np.float32)
    v = np.random.rand(n_maxlen+6,n_maxlen+6).astype(np.float32)
    w = np.random.randint(0,n_maxlen,size=2) 
    
    chainer2onnx.generate_testcase(model, [u,v,w])
