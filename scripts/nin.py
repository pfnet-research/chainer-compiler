import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L


class NIN(chainer.Chain):

    """Network-in-Network example model."""

    insize = 227

    def __init__(self, compute_accuracy=False):
        super(NIN, self).__init__()
        self.compute_accuracy = compute_accuracy
        conv_init = I.HeNormal()  # MSRA scaling

        with self.init_scope():
            self.mlpconv1 = L.MLPConvolution2D(
                None, (96, 96, 96), 11, stride=4, conv_init=conv_init)
            self.mlpconv2 = L.MLPConvolution2D(
                None, (256, 256, 256), 5, pad=2, conv_init=conv_init)
            self.mlpconv3 = L.MLPConvolution2D(
                None, (384, 384, 384), 3, pad=1, conv_init=conv_init)
            self.mlpconv4 = L.MLPConvolution2D(
                None, (1024, 1024, 1000), 3, pad=1, conv_init=conv_init)

    def forward(self, x, t):
        h = F.max_pooling_2d(F.relu(self.mlpconv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv2(h)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv3(h)), 3, stride=2)
        h = self.mlpconv4(F.dropout(h))
        h = F.reshape(F.average_pooling_2d(h, 6), (len(x), 1000))

        #loss = F.softmax_cross_entropy(h, t)
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
        batch_size = chainer.Variable(np.array(t.size, np.float32),
                                      name='batch_size')
        self.extra_inputs = [batch_size]
        loss = -log_prob / batch_size
        loss.name = 'loss'
        return loss
