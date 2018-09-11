# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F

# Network definition

    

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs, ilens):
        #print(xs)
        xs = 
        xs = F.pad_sequence(xs)
        return xs

# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(12)

    n_batch = 5
    n_maxlen = 10
    n_in = 3

    model = A()
    
    w = np.random.randint(1,n_maxlen,size=n_batch)
    v = [ np.random.rand(x,n_in).astype(np.float32) for x in w]
     
    v = F.pad_sequence(v)
    chainer2onnx.generate_testcase(model, [v,w])
