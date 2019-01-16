import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self, compute_accuracy=False):
        super(Alex, self).__init__()
        self.compute_accuracy = compute_accuracy
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

    def forward(self, x, t):
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

        # EDIT(hamaji): ONNX-chainer cannot output SoftmaxCrossEntropy.
        # loss = F.softmax_cross_entropy(h, t)
        loss = self.softmax_cross_entropy(h, t)
        if self.compute_accuracy:
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
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
        batch_size = chainer.Variable(np.array(t.shape[0], np.float32),
                                      name='batch_size')
        self.extra_inputs = [batch_size]
        loss = -log_prob / batch_size
        loss.name = 'loss'
        return loss


class AlexFp16(Alex):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        chainer.Chain.__init__(self)
        self.dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=4,
                                         initialW=W, initial_bias=bias)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2,
                                         initialW=W, initial_bias=bias)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.fc6 = L.Linear(None, 4096, initialW=W, initial_bias=bias)
            self.fc7 = L.Linear(None, 4096, initialW=W, initial_bias=bias)
            self.fc8 = L.Linear(None, 1000, initialW=W, initial_bias=bias)

    def forward(self, x, t):
        return Alex.forward(self, F.cast(x, self.dtype), t)
