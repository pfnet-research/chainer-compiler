import os
import numpy as np

current_id = 0

slice_int_max = 2 ** 31 - 1

def get_guid():
    global current_id
    id = current_id
    current_id += 1
    return id

def reset_guid():
    global current_id
    current_id = 0

def numpy_type_2_int(t):
    if t == np.int32:
        return 0
    if t == np.float32:
        return 1
    assert(False)

def int_2_numpy_type(n):
    if n == 0:
        return np.int32
    if n == 1:
        return np.float32
    assert(False)

def create_obj_value_name_with_constant(value):
    return '@C_' + str(value)

def create_obj_value_name_with_attribute(name : "str", pre_name : "str"):
    if len(pre_name) > 0 and pre_name[0] != '@':
        return pre_name
    else:
        return name

def clip_head(s : 'str'):
    s = s.split('\n')
    # print(s)
    hs = os.path.commonprefix(list(filter(lambda x: x != '', s)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    s = map(lambda x: x[ls:], s)
    return '\n'.join(s)

class LineProperty():
    def __init__(self, lineno = -1, filename = ''):
        self.lineno = lineno
        self.filename = filename

    def get_line_str(self) -> 'str':
        return 'L.' + str(self.lineno)

    def __str__(self):

        if self.filename == '':
            return 'L.' + str(self.lineno)

        return self.filename + '[L.' + str(self.lineno) + ']'
