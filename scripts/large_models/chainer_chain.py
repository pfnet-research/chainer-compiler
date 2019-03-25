import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


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


def get_chainer_model(chainer_chain, dtype, key):
    batchsize = 4
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype(dtype)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    model = Wrapper(chainer_chain(pretrained_model=None), key)
    return model, [x, t]


def get_resnet50(dtype=None):
    return get_chainer_model(L.ResNet50Layers, dtype, 'fc6')


def get_resnet152(dtype=None):
    return get_chainer_model(L.ResNet152Layers, dtype, 'fc6')


def get_vgg16(dtype=None):
    return get_chainer_model(L.VGG16Layers, dtype, 'fc8')


def get_vgg19(dtype=None):
    return get_chainer_model(L.VGG19Layers, dtype, 'fc8')
