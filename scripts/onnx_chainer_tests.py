import glob
import os

from test_case import TestCase


def get(target_opsets=[]):
    tests = []

    if len(target_opsets) is 0:
        target_opsets = [10]

    for opset in target_opsets:
        for test_dir in sorted(glob.glob('out/opset{}/*'.format(opset))):
            name = 'onnx_chainer_{}_{}'.format(
                os.path.basename(test_dir), opset)
            fail = ('dilated' in name or
                    'hard_sigmoid' in name or
                    'pad_edge' in name or
                    'pad_reflect' in name or
                    'prod_axis' in name or
                    'roipooling2d' in name or
                    'prelu' in name or
                    'tile' in name or
                    'resizeimages' in name)
            equal_nan = ('powvarvar' in name or
                         'arcsin' in name or
                         'arccos' in name)
            tests.append(TestCase(name=name,
                                  test_dir=test_dir,
                                  equal_nan=equal_nan,
                                  fail=fail,
                                  opset_version=opset))

    return tests
