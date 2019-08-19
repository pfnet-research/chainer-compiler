import chainer.functions as F
import numpy as np
import inspect

def check_attribute_value(actual_value, expected_value, func_name='', arg_name=''):
    return

def check_attribute_scalar(value, func_name='', arg_name=''):
    return

def chainer_clipped_relu(x, z=20.0):
    return F.minimum(F.maximum(0.0, x), z)

def numpy_clip(a, a_min, a_max, out=None):
    check_attribute_scalar(a_min,'numpy.clip', 'a_min')
    check_attribute_scalar(a_max,'numpy.clip', 'a_max')
    check_attribute_value(out, None, 'numpy.clip', 'out')

    a = F.clip(a, a_min, a_max)
    return a

def check_args(func1, func2):
    sig1 = inspect.signature(func1)
    sig2 = inspect.signature(func2)
    assert(len(sig1.parameters) == len(sig2.parameters))
    values1 = list(sig1.parameters.values())
    values2 = list(sig2.parameters.values())

    for i in range(len(sig1.parameters.keys())):
        assert(values1[i].name == values2[i].name)
        assert(values1[i].default == values2[i].default)


check_args(F.clipped_relu, chainer_clipped_relu)
