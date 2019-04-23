# coding: utf-8

import chainer
import chainer.functions as F
import testtools

class ResizeImages(chainer.Chain):
    def forward(self, x):
        y1 = F.resize_images(x, (257, 513))
        return y1


# ======================================

import numpy as np

if __name__ == '__main__':
    x = np.random.rand(1, 1, 129, 257).astype(np.float32)
    # x = np.random.rand(1, 256, 129, 257).astype(np.float32)
    testtools.generate_testcase(ResizeImages, [x])
