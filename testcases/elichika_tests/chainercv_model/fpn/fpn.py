import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainer_compiler.elichika.parser import flags

class FPN(chainer.Chain):
    """An extractor class of Feature Pyramid Networks.

    This class wraps a feature extractor and provides
    multi-scale features.

    Args:
        base (Link): A base feature extractor.
            It should have :meth:`forward` and :obj:`mean`.
            :meth:`forward` should take a batch of images and return
            feature maps of them. The size of the :math:`k+1`-th feature map
            should be the half as that of the :math:`k`-th feature map.
        n_base_output (int): The number of feature maps
            that :obj:`base` returns.
        scales (tuple of floats): The scales of feature maps.

    """

    def __init__(self, base, n_base_output, scales):
        super(FPN, self).__init__()
        with self.init_scope():
            self.base = base
            self.inner = chainer.ChainList()
            self.outer = chainer.ChainList()

        init = {'initialW': initializers.GlorotNormal()}
        for _ in range(n_base_output):
            self.inner.append(L.Convolution2D(256, 1, **init))
            self.outer.append(L.Convolution2D(256, 3, pad=1, **init))

        self.scales = scales
        # hacks
        self.n_base_output = n_base_output
        self.n_base_output_minus1 = n_base_output - 1
        self.scales_minus_n_base_output = len(scales) - n_base_output

    @property
    def mean(self):
        return self.base.mean

    def forward(self, x):
        hs = self.base(x)

        with flags.for_unroll():
            for i in range(self.n_base_output_minus1, -1, -1):
                hs[i] = self.inner[i](hs[i])
                if i < self.n_base_output_minus1:
                    hs[i] += F.unpooling_2d(hs[i + 1], 2, cover_all=False)

            for i in range(self.n_base_output):
                hs[i] = self.outer[i](hs[i])

            for _ in range(self.scales_minus_n_base_output):
                hs.append(F.max_pooling_2d(hs[-1], 1, stride=2, cover_all=False))

        return hs


# ======================================

from chainer_compiler.elichika import testtools
import numpy as np

def main():
    np.random.seed(314)

    base = lambda x: x
    n_base_output = 4
    scales = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
    model = FPN(base=base, n_base_output=n_base_output, scales=scales)

    bsize = 2
    np.seed = 1

    v = [
        np.random.rand(bsize, 256, 56, 56).astype(np.float32),
        np.random.rand(bsize, 512, 28, 28).astype(np.float32),
        np.random.rand(bsize, 1024, 14, 14).astype(np.float32),
        np.random.rand(bsize, 2048, 7, 7).astype(np.float32),
    ]
    testtools.generate_testcase(model, [v])

if __name__ == '__main__':
    main()