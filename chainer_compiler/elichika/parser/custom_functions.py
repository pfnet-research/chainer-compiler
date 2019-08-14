import chainer.functions as F
import numpy as np

def check_attribute_value(actual_value, expected_value, func_name=''):
    return

def check_attribute_scalar(value, func_name=''):
    return

def numpy_clip(a, a_min, a_max, out=None):
    check_attribute_scalar(a_min,'numpy.clip')
    check_attribute_scalar(a_max,'numpy.clip')
    check_attribute_value(out, None, 'numpy.clip')

    a = F.clip(a, a_min, a_max)
    return a
