import glob
import os

from test_case import TestCase


def get():
    tests = []

    for test_dir in sorted(glob.glob('out/opset10/*')):
        name = 'onnx_chainer_' + os.path.basename(test_dir)
        fail = ('dilated' in name or
                'hard_sigmoid' in name or
                'pad_edge' in name or
                'pad_reflect' in name or
                'prod' in name or
                'roipooling2d' in name or
                'prelu' in name or
                'tile' in name or
                'group3' in name or
                'resizeimages' in name)
        tests.append(TestCase(name=name,
                              test_dir=test_dir,
                              fail=fail))

    return tests
