# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2, groups=1):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True,
                groups=groups)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch, groups=1):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True,
                groups=groups)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2, groups=1):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride, groups))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch, groups))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):

    insize = 224

    def __init__(self, compute_accuracy=False):
        super(ResNet50, self).__init__()
        self.compute_accuracy = compute_accuracy
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(3, 64, 64, 256, 1)
            self.res3 = Block(4, 256, 128, 512)
            self.res4 = Block(6, 512, 256, 1024)
            self.res5 = Block(3, 1024, 512, 2048)
            self.fc = L.Linear(2048, 1000)

    def forward(self, x, t):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        #loss = F.softmax_cross_entropy(h, t)
        loss = self.softmax_cross_entropy(h, t)
        if self.compute_accuracy:
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, np.argmax(t, axis=1))}, self)
        else:
            chainer.report({'loss': loss}, self)
        return loss

    def softmax_cross_entropy(self, y, t):
        import numpy as np

        log_softmax = F.log_softmax(y)
        # SelectItem is not supported by onnx-chainer.
        # TODO(hamaji): Support it?
        # log_prob = F.select_item(log_softmax, t)

        # TODO(hamaji): Currently, F.sum with axis=1 cannot be
        # backpropped properly.
        # log_prob = F.sum(log_softmax * t, axis=1)
        # self.batch_size = chainer.Variable(np.array(t.size, np.float32),
        #                                    name='batch_size')
        # return -F.sum(log_prob, axis=0) / self.batch_size
        log_prob = F.sum(log_softmax * t, axis=(0, 1))
        batch_size = chainer.Variable(np.array(t.size, np.float32),
                                      name='batch_size')
        self.extra_inputs = [batch_size]
        loss = -log_prob / batch_size
        loss.name = 'loss'
        return loss
