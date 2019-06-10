# coding: utf-8

import argparse
import pickle
import sys
import os

import chainer
import chainer.functions as F
from chainer.backends import cuda


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
        return self.fn(x, rois, roi_indices, 7,
                       0.25, 2)


class FPN_ROIAlign2D_1st_scale(chainer.Chain):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.outsize = 7
        self.spatial_scale = 1 / 4
        self.sampling_ratio = 2

    def forward(self, x, rois, roi_indices):
        return self.fn(x, rois, roi_indices, 7,
                       0.25, 2)


class FPN_ROIAlign2D_2nd_scale(chainer.Chain):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.outsize = 7
        self.spatial_scale = 1 / 8
        self.sampling_ratio = 2

    def forward(self, x, rois, roi_indices):
        return self.fn(x, rois, roi_indices, 7,
                       0.125, 2)


class FPN_ROIAlign2D_3rd_scale(chainer.Chain):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.outsize = 7
        self.spatial_scale = 1 / 16
        self.sampling_ratio = 2

    def forward(self, x, rois, roi_indices):
        return self.fn(x, rois, roi_indices, 7,
                       0.0625, 2)


class FPN_ROIAlign2D_4th_scale(chainer.Chain):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.outsize = 7
        self.spatial_scale = 1 / 32
        self.sampling_ratio = 2

    def forward(self, x, rois, roi_indices):
        return self.fn(x, rois, roi_indices, 7,
                       0.03125, 2)


class FPN_ROIAlign2D_5th_scale(chainer.Chain):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.outsize = 7
        self.spatial_scale = 1 / 64
        self.sampling_ratio = 2

    def forward(self, x, rois, roi_indices):
        return self.fn(x, rois, roi_indices, 7,
                       0.015625, 2)


# ======================================

from chainer_compiler import ch2o

if __name__ == '__main__':
    import numpy as np

    if len(sys.argv) > 1 and sys.argv[1].startswith('-'):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--data',
            type=str,
            default=None,
            help='pickle data which contains hs, rois and roi_indices')
        parser.add_argument(
            '--index',
            type=int,
            default=0,
            help='which h in hs is used')
        parser.add_argument(
            '--out',
            type=str,
            default=".",
            help='')
        parser.add_argument('--gpu', action='store_true')
        args = parser.parse_args()

        print(args.data)
        with open(args.data, "rb") as f:
            data = pickle.load(f)
        hs = [data_var.array for data_var in data["hs"]]
        rois = data["rois"]
        roi_indices = data["roi_indices"]
        if args.gpu:
            hs = [chainer.cuda.to_gpu(h) for h in hs]
            rois = chainer.cuda.to_gpu(rois)
            roi_indices = chainer.cuda.to_gpu(roi_indices)
        assert(len(hs) == 5)
        assert(len(rois) == 5)
        assert(len(roi_indices) == 5)
        for i, FPN_ROIAlign2D in enumerate((FPN_ROIAlign2D_1st_scale,
                                            FPN_ROIAlign2D_2nd_scale,
                                            FPN_ROIAlign2D_3rd_scale,
                                            FPN_ROIAlign2D_4th_scale,
                                            FPN_ROIAlign2D_5th_scale)):
            ch2o.generate_testcase(
                FPN_ROIAlign2D(F.roi_average_align_2d),
                [hs[i], rois[i], roi_indices[i]],
                output_dir=os.path.join(args.out, "fpn_roi_align_2d_pyramid{}_scale".format(i)),
                use_gpu=args.gpu)
    else:
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
