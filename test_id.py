# coding: utf-8

import chainer
import chainer.functions as F

import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(10)
    def forward(self, x):
        #a = F.relu(x)
        #b = F.relu(a)
        #return a,b
        x = self.l1(x)
        x = F.relu(x)
        return x


# ======================================

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    import testcasegen

    model = A()

    v = np.random.rand(5,3).astype(np.float32)
    import test_mxnet
    test_mxnet.check_compatibility(model, v)
    # testcasegen.generate_testcase(model, v)
    
    # import test_mxnet
    # test_mxnet.check_compatibility(model, v)
