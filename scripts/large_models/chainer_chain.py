import collections

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.functions.noise.dropout import dropout
from chainer.utils import argument


class Wrapper(chainer.Chain):

    def __init__(self, predictor, key=None):
        super().__init__()

        with self.init_scope():
            self.predictor = predictor
        self.key = key

    def __call__(self, x, t):
        if self.key is None:
            y = self.predictor(x)
        else:
            y = self.predictor(x, layers=[self.key])[self.key]
        y = F.softmax_cross_entropy(y, t)
        return y


# Code is copied from chainer.links.model.
class VGGLayers(chainer.Chain):

    def __init__(self, pretrained_model='auto', n_layers=16):
        super(VGGLayers, self).__init__()
        kwargs = {}

        if n_layers not in [16, 19]:
            raise ValueError(
                'The n_layers argument should be either 16 or 19,'
                'but {} was given.'.format(n_layers)
            )

        with self.init_scope():
            self.conv1_1 = Convolution2D(3, 64, 3, 1, 1, **kwargs)
            self.conv1_2 = Convolution2D(64, 64, 3, 1, 1, **kwargs)
            self.conv2_1 = Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = Convolution2D(128, 128, 3, 1, 1, **kwargs)
            self.conv3_1 = Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv4_1 = Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_1 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.fc6 = Linear(512 * 7 * 7, 4096, **kwargs)
            self.fc7 = Linear(4096, 4096, **kwargs)
            self.fc8 = Linear(4096, 1000, **kwargs)
            if n_layers == 19:
                self.conv3_4 = Convolution2D(256, 256, 3, 1, 1, **kwargs)
                self.conv4_4 = Convolution2D(512, 512, 3, 1, 1, **kwargs)
                self.conv5_4 = Convolution2D(512, 512, 3, 1, 1, **kwargs)

    @property
    def functions(self):
        # This class will not be used directly.
        raise NotImplementedError

    @property
    def available_layers(self):
        return list(self.functions.keys())

    def forward(self, x, layers=None, **kwargs):
        if layers is None:
            layers = ['prob']

        if kwargs:
            argument.check_unexpected_kwargs(
                kwargs, test='test argument is not supported anymore. '
                'Use chainer.using_config'
            )
            argument.assert_kwargs_empty(kwargs)

        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)


class VGG16Layers(VGGLayers):

    def __init__(self, pretrained_model='auto'):
        super(VGG16Layers, self).__init__(pretrained_model, 16)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1_1', [self.conv1_1, relu]),
            ('conv1_2', [self.conv1_2, relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, relu]),
            ('conv2_2', [self.conv2_2, relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, relu]),
            ('conv3_2', [self.conv3_2, relu]),
            ('conv3_3', [self.conv3_3, relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, relu]),
            ('conv4_2', [self.conv4_2, relu]),
            ('conv4_3', [self.conv4_3, relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, relu]),
            ('conv5_2', [self.conv5_2, relu]),
            ('conv5_3', [self.conv5_3, relu]),
            ('pool5', [_max_pooling_2d]),
            # ('fc6', [self.fc6, relu, dropout]),
            ('fc6', [self.fc6, relu, lambda x: dropout(x, ratio=0.0)]),
            # ('fc7', [self.fc7, relu, dropout]),
            ('fc7', [self.fc7, relu, lambda x: dropout(x, ratio=0.0)]),
            ('fc8', [self.fc8]),
            ('prob', [softmax]),
        ])


class VGG19Layers(VGGLayers):

    def __init__(self, pretrained_model='auto'):
        super(VGG19Layers, self).__init__(pretrained_model, 19)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1_1', [self.conv1_1, relu]),
            ('conv1_2', [self.conv1_2, relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, relu]),
            ('conv2_2', [self.conv2_2, relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, relu]),
            ('conv3_2', [self.conv3_2, relu]),
            ('conv3_3', [self.conv3_3, relu]),
            ('conv3_4', [self.conv3_4, relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, relu]),
            ('conv4_2', [self.conv4_2, relu]),
            ('conv4_3', [self.conv4_3, relu]),
            ('conv4_4', [self.conv4_4, relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, relu]),
            ('conv5_2', [self.conv5_2, relu]),
            ('conv5_3', [self.conv5_3, relu]),
            ('conv5_4', [self.conv5_4, relu]),
            ('pool5', [_max_pooling_2d]),
            # ('fc6', [self.fc6, relu, dropout]),
            ('fc6', [self.fc6, relu, lambda x: dropout(x, ratio=0.0)]),
            # ('fc7', [self.fc7, relu, dropout]),
            ('fc7', [self.fc7, relu, lambda x: dropout(x, ratio=0.0)]),
            ('fc8', [self.fc8]),
            ('prob', [softmax]),
        ])


def get_chainer_model(chainer_chain, dtype, key):
    batchsize = 3
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype(dtype)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    model = Wrapper(chainer_chain(pretrained_model=None), key)
    return model, [x, t]


def get_resnet50(dtype=None):
    return get_chainer_model(L.ResNet50Layers, dtype, 'fc6')


def get_resnet152(dtype=None):
    return get_chainer_model(L.ResNet152Layers, dtype, 'fc6')


def get_vgg16(dtype=None):
    return get_chainer_model(VGG16Layers, dtype, 'fc8')


def get_vgg19(dtype=None):
    return get_chainer_model(VGG19Layers, dtype, 'fc8')
