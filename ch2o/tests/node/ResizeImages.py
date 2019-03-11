# coding: utf-8

import chainer
import chainer.functions as F


class ResizeImages(chainer.Chain):
    def forward(self, x):
        y1 = F.resize_images(x, (257, 513))
        return y1


# ======================================

import ch2o
import numpy as np

if __name__ == '__main__':
    x = np.random.rand(1, 256, 129, 257).astype(np.float32)
    ch2o.generate_testcase(ResizeImages, [x])
