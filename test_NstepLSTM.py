# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self,n_layer,n_in,n_out):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepLSTM(n_layer,n_in,n_out,0.1)

    def forward(self, x):
        hy,cs,ys = self.l1(None,None,x)
        return hy,cs,ys


# ======================================

import testcasegen


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    
    #とりあえずというかんじ 
    # これはなにかのtestになっているのだろうか
    
    layn = 1
    model = A(layn,3,5)
    
    x = [np.random.rand(4, 3).astype(np.float32) for _ in range(2)]
    x = [x]
    testcasegen.generate_testcase(model, x)
    
