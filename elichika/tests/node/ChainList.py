# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import testtools

class LinearBlockCall(chainer.Chain):
    def __init__(self):
        super(LinearBlockCall, self).__init__()
        with self.init_scope():
            self.l = L.Linear(10, 10)

    def __call__(self, x):
        return F.relu(self.l(x))

class LinearBlock(chainer.Chain):
    def __init__(self):
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.l = L.Linear(10, 10)

    def forward(self, x):
        return F.relu(self.l(x))

class LinearChainList(chainer.ChainList):

    def __init__(self):
        super(LinearChainList, self).__init__(
            LinearBlock(),
            LinearBlock(),
        )

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x

class LinearCallChainList(chainer.ChainList):

    def __init__(self):
        super(LinearCallChainList, self).__init__(
            LinearBlockCall(),
            LinearBlockCall(),
        )

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x

def main():
    x = np.random.rand(5, 10).astype(np.float32)
    testtools.generate_testcase(LinearChainList(), [x], subname='chain')
    testtools.generate_testcase(LinearCallChainList(), [x], subname='chain_call')

if __name__ == '__main__':
    main()
