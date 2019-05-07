"""Tests for ChainerCV related custom ops."""


import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import onnx

import onnx_script
import test_case

_has_chnainercv = True

try:
    import chainercv_rpn
except ImportError:
    _has_chnainercv = False


def aranges(*shape):
    r = np.prod(shape)
    v = np.arange(r).reshape(shape).astype(np.float32)
    v -= r / 2 + 0.1
    return v


def _get_scales():
    return (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)


def _get_hs(num_channels):
    hs = []
    for h, w in [(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)]:
        hs.append(aranges(1, num_channels, h, w))
    return hs


def _get_rpn_locs_confs():
    locs = []
    confs = []
    for i in [163200, 40800, 10200, 2550, 663]:
        locs.append(aranges(1, i, 4))
        confs.append(aranges(1, i))
    return locs, confs


def chainercv_test_rpn_decode(test_name):
    rpn = chainercv_rpn.RPN(_get_scales())
    hs = _get_hs(1)
    locs, confs = _get_rpn_locs_confs()
    anchors = rpn.anchors(h.shape[2:] for h in hs)
    in_shape = (1, 3, 800, 1088)
    rois, roi_indices = rpn.decode(
        [chainer.Variable(l) for l in locs],
        [chainer.Variable(c) for c in confs],
        anchors, in_shape)

    gb = onnx_script.GraphBuilder(test_name)
    hs_v = [gb.input('hs_%d' % i, h) for i, h in enumerate(hs)]
    locs_v = [gb.input('loc_%d' % i, l) for i, l in enumerate(locs)]
    confs_v = [gb.input('conf_%d' % i, c) for i, c in enumerate(confs)]
    in_shape_v = gb.input('in_shape', np.array(in_shape))

    rois_v = 'rois'
    roi_indices_v = 'roi_indices'
    gb.ChainerDoSomething(hs_v + locs_v + confs_v + [in_shape_v],
                          outputs=[rois_v, roi_indices_v],
                          function_name='ChainerCVRPNDecode')
    gb.output(rois_v, rois)
    gb.output(roi_indices_v, roi_indices)

    gb.gen_test()


class TestCase(test_case.TestCase):
    def __init__(self, name, func, **kwargs):
        super(TestCase, self).__init__('out', name, **kwargs)
        self.func = func


def get_tests():
    if not _has_chnainercv:
        return []

    tests = []
    def test(name, func, **kwargs):
        tests.append(TestCase(name, func, **kwargs))

    test('chainercv_test_rpn_decode', chainercv_test_rpn_decode)

    return tests
