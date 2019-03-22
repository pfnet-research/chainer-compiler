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


def get_resnet50(dtype=None):
    batchsize = 4
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype(dtype)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    model = Wrapper(L.ResNet50Layers(pretrained_model=None),
                    'fc6')
    return model, [x, t]
