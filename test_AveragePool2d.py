import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class A(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        x = F.max_pooling_2d(x,(1,3), stride=(1,4),pad=(0,1))
        return x

# from https://github.com/chainer/chainer/blob/master/examples/imagenet/alex.py


if __name__ == '__main__':
    np.random.seed(314)

    model = A()

    # batch * channel * H * W
    v = np.random.rand(2, 3, 1, 13).astype(np.float32)

    import testcasegen
    testcasegen.generate_testcase(model, [v])
