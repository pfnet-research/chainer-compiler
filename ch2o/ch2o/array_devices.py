import numpy as np


xps = [np]

try:
    import cupy
    xps.append(cupy)
except:
    pass

try:
    import chainerx
    xps.append(chainerx)
except:
    pass


def get_array_devices():
    return xps
