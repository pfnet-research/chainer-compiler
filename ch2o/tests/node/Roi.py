# coding: utf-8

import chainer
import chainer.functions as F


class ROIPool2D(chainer.Chain):
    def __init__(self, fn, outsize, spatial_scale):
        super(ROIPool2D, self).__init__()
        self.fn = fn
        self.outsize = outsize
        self.spatial_scale = spatial_scale

    def forward(self, x, rois, roi_indices):
        return self.fn(x, rois, roi_indices, 7, 1.2)


class ROIAlign2D(chainer.Chain):
    def __init__(self, fn, outsize, spatial_scale, sampling_ratio):
        super(ROIAlign2D, self).__init__()
        self.fn = fn
        self.outsize = outsize
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, x, rois, roi_indices):
        return self.fn(x, rois, roi_indices, 7, 1.2, 2)


# ======================================

import ch2o

if __name__ == '__main__':
    import numpy as np

    x = np.arange(2 * 3 * 5 * 5).reshape((2, 3, 5, 5)).astype(np.float32)
    rois = np.array([[0, 1, 3, 4], [1, 0.3, 4, 2.6]]).astype(np.float32)
    roi_indices = np.array([0, 1]).astype(np.int32)

    ch2o.generate_testcase(ROIPool2D(F.roi_max_pooling_2d, 7, 1.2),
                           [x, rois, roi_indices],
                           subname='max_pool')
    ch2o.generate_testcase(ROIPool2D(F.roi_average_pooling_2d, 7, 1.2),
                           [x, rois, roi_indices],
                           subname='avg_pool')
    ch2o.generate_testcase(ROIAlign2D(F.roi_max_align_2d, 7, 1.2, 2),
                           [x, rois, roi_indices],
                           subname='max_align')
    ch2o.generate_testcase(ROIAlign2D(F.roi_average_align_2d, 7, 1.2, 3),
                           [x, rois, roi_indices],
                           subname='avg_align')
