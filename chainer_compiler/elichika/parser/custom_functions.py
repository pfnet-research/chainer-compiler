import chainer.functions as F
import numpy as np

def check_attribute_value(actual_value, expected_value, func_name='', arg_name=''):
    return

def check_attribute_scalar(value, func_name='', arg_name=''):
    return

def numpy_clip(a, a_min, a_max, out=None):
    check_attribute_scalar(a_min,'numpy.clip', 'a_min')
    check_attribute_scalar(a_max,'numpy.clip', 'a_max')
    check_attribute_value(out, None, 'numpy.clip', 'out')

    a = F.clip(a, a_min, a_max)
    return a
