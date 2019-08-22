import chainer
import chainer.functions as F
import chainer.links as L

from chainer.functions.activation import relu
from chainer.functions.array import concat
from chainer import link
from chainer.links.connection import convolution_2d


class Inception(link.Chain):

    """Inception module of GoogLeNet.
    It applies four different functions to the input array and concatenates
    their outputs along the channel dimension. Three of them are 2D
    convolutions of sizes 1x1, 3x3 and 5x5. Convolution paths of 3x3 and 5x5
    sizes have 1x1 convolutions (called projections) ahead of them. The other
    path consists of 1x1 convolution (projection) and 3x3 max pooling.
    The output array has the same spatial size as the input. In order to
    satisfy this, Inception module uses appropriate padding for each
    convolution and pooling.
    See: `Going Deeper with Convolutions <https://arxiv.org/abs/1409.4842>`_.
    Args:
        in_channels (int or None): Number of channels of input arrays.
        out1 (int): Output size of 1x1 convolution path.
        proj3 (int): Projection size of 3x3 convolution path.
        out3 (int): Output size of 3x3 convolution path.
        proj5 (int): Projection size of 5x5 convolution path.
        out5 (int): Output size of 5x5 convolution path.
        proj_pool (int): Projection size of max pooling path.
        conv_init (:ref:`initializer <initializer>`): Initializer to
            initialize the convolution matrix weights.
            When it is :class:`numpy.ndarray`, its ``ndim`` should be 4.
        bias_init (:ref:`initializer <initializer>`): Initializer to
            initialize the convolution matrix weights.
            When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
    """

    def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool,
                 conv_init=None, bias_init=None):
        super(Inception, self).__init__()
        with self.init_scope():
            self.conv1 = convolution_2d.Convolution2D(
                in_channels, out1, 1, initialW=conv_init,
                initial_bias=bias_init)
            self.proj3 = convolution_2d.Convolution2D(
                in_channels, proj3, 1, initialW=conv_init,
                initial_bias=bias_init)
            self.conv3 = convolution_2d.Convolution2D(
                proj3, out3, 3, pad=1, initialW=conv_init,
                initial_bias=bias_init)
            self.proj5 = convolution_2d.Convolution2D(
                in_channels, proj5, 1, initialW=conv_init,
                initial_bias=bias_init)
            self.conv5 = convolution_2d.Convolution2D(
                proj5, out5, 5, pad=2, initialW=conv_init,
                initial_bias=bias_init)
            self.projp = convolution_2d.Convolution2D(
                in_channels, proj_pool, 1, initialW=conv_init,
                initial_bias=bias_init)

    def forward(self, x):
        """Computes the output of the Inception module.
        Args:
            x (~chainer.Variable): Input variable.
        Returns:
            Variable: Output variable. Its array has the same spatial size and
            the same minibatch size as the input array. The channel dimension
            has size ``out1 + out3 + out5 + proj_pool``.
        """
        out1 = self.conv1(x)
        out3 = self.conv3(relu.relu(self.proj3(x)))
        out5 = self.conv5(relu.relu(self.proj5(x)))
        pool = self.projp(F.max_pooling_2d(
            x, 3, stride=1, pad=1))

        y = relu.relu(concat.concat((out1, out3, out5, pool), axis=1))
        return y


class GoogLeNet(chainer.Chain):

    insize = 224

    def __init__(self):
        super(GoogLeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  64, 7, stride=2, pad=3)
            self.conv2_reduce = L.Convolution2D(None,  64, 1)
            self.conv2 = L.Convolution2D(None, 192, 3, stride=1, pad=1)
            # 以下、L.Inceptionを上で定義したInceptionとする
            self.inc3a = Inception(None,  64,  96, 128, 16,  32,  32)
            self.inc3b = Inception(None, 128, 128, 192, 32,  96,  64)
            self.inc4a = Inception(None, 192,  96, 208, 16,  48,  64)
            self.inc4b = Inception(None, 160, 112, 224, 24,  64,  64)
            self.inc4c = Inception(None, 128, 128, 256, 24,  64,  64)
            self.inc4d = Inception(None, 112, 144, 288, 32,  64,  64)
            self.inc4e = Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5a = Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5b = Inception(None, 384, 192, 384, 48, 128, 128)
            self.loss3_fc = L.Linear(None, 1000)

            self.loss1_conv = L.Convolution2D(None, 128, 1)
            self.loss1_fc1 = L.Linear(None, 1024)
            self.loss1_fc2 = L.Linear(None, 1000)

            self.loss2_conv = L.Convolution2D(None, 128, 1)
            self.loss2_fc1 = L.Linear(None, 1024)
            self.loss2_fc2 = L.Linear(None, 1000)

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        # return h
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss1_conv(l))
        l = F.relu(self.loss1_fc1(l))
        l = self.loss1_fc2(l)
        loss1 = F.softmax_cross_entropy(l, t)
        # return loss1
        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss2_conv(l))
        l = F.relu(self.loss2_fc1(l))
        l = self.loss2_fc2(l)
        loss2 = F.softmax_cross_entropy(l, t)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4))
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        # accuracy = F.accuracy(h, t)

        """
        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)
        """
        return loss

# from https://github.com/chainer/chainer/blob/master/examples/imagenet/googlenet.py


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = GoogLeNet()  # 224
    v = np.random.rand(2, 3, 227, 227).astype(np.float32)
    t = np.random.randint(1000, size=2)

    from chainer_compiler import ch2o
    ch2o.generate_testcase(model, [v, t])
