import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self, shrink_ratio=1):
        super(Alex, self).__init__()
        sr = shrink_ratio
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96 // sr, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256 // sr,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384 // sr,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384 // sr,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256 // sr,  3, pad=1)
            self.fc6 = L.Linear(None, 4096 // sr)
            self.fc7 = L.Linear(None, 4096 // sr)
            self.fc8 = L.Linear(None, 1000)

    def forward(self, x, t):
        # def forward(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        #loss = h

        # chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

# from https://github.com/chainer/chainer/blob/master/examples/imagenet/alex.py


if __name__ == '__main__':
    np.random.seed(314)

    model = Alex(shrink_ratio=23)

    # batch * channel * H * W
    v = np.random.rand(2, 3, 227, 227).astype(np.float32)
    w = np.random.randint(1000, size=2)

    import chainer_compiler.ch2o
    ch2o.generate_testcase(model, [v, w])
