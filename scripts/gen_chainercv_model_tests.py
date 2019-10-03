"""Tests for ChainerCV models."""

import os
import shutil

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import onnx
import onnx_chainer

import test_case

_has_chnainercv = True

try:
    import chainercv
    import chainercv.links as CV
except ImportError:
    _has_chnainercv = False


def chainercv_model_test(model):
    def fn(test_name):
        np.random.seed(42)
        x = np.random.rand(1, 3, 224, 224).astype(np.float32)
        test_dir = os.path.join('out', test_name)
        shutil.rmtree(test_dir, ignore_errors=True)
        onnx_chainer.export_testcase(model, [x],
                                     test_dir,
                                     opset_version=9)

    return fn


def get_tests():
    if not _has_chnainercv:
        return []

    tests = []
    def test(name, func, **kwargs):
        case = test_case.TestCase('out', name, **kwargs)
        case.func = func
        tests.append(case)

    np.random.seed(42)
    test('chainercv_test_yolo_v2_tiny',
         chainercv_model_test(chainercv.experimental.links.YOLOv2Tiny(1000)))

    return tests


def main():
    for test in get_tests():
        test.func(test.name)


if __name__ == '__main__':
    main()
