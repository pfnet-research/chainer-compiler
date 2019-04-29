# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import testtools

class LinearBlock(chainer.Chain):
    def __init__(self):
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.l = L.Linear(10, 10)

    # TODO (durswd): support __call__ without adhoc codes            
    #def __call__(self, x):
    def forward(self, x):
        return F.relu(self.l(x))

class A(chainer.ChainList):

    def __init__(self):
        super(A, self).__init__(
            LinearBlock(),
            LinearBlock(),
        )

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x


def main():
    x = np.random.rand(5, 10).astype(np.float32)
    testtools.generate_testcase(A(), [x])

if __name__ == '__main__':
    main()
