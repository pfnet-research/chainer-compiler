import os
import numpy as np
from elichika.parser import config

current_id = 0

slice_int_max = 2 ** 31 - 1

dtype_float32 = np.array(1.0, dtype=np.float32).dtype
dtype_float64 = np.array(1.0, dtype=np.float64).dtype
dtype_int = np.array(1.0, dtype=np.int).dtype

def get_guid():
    global current_id
    id = current_id
    current_id += 1
    return id


def reset_guid():
    global current_id
    current_id = 0

def print_warning(s, lineprop):
    print('warning : {} in {}'.format(s, lineprop))

def is_disabled_module(m):
    return m in config.disabled_modules

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

def create_obj_value_name_with_attribute(name: "str", pre_name: "str"):
    if len(pre_name) > 0 and pre_name[0] != '@':
        return pre_name
    else:
        return name

def clip_head(s: 'str'):
    splitted = s.split('\n')
    
    # remove comments
    comment_count = 0
    indent_targets = []
    for sp in splitted:
        if '"""' in sp or "'''" in sp:
            comment_count += 1
        else:
            if comment_count % 2 == 0:
                indent_targets.append(sp)

    hs = os.path.commonprefix(list(filter(lambda x: x != '', indent_targets)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    strs = map(lambda x: x[ls:], splitted)
    return '\n'.join(strs)


class LineProperty():
    def __init__(self, lineno=-1, filename=''):
        self.lineno = lineno
        self.filename = filename

    def get_line_str(self) -> 'str':
        return 'L.' + str(self.lineno)

    def __str__(self):

        if self.filename == '':
            return 'L.' + str(self.lineno)

        return self.filename + '[L.' + str(self.lineno) + ']'

class UnimplementedError(Exception):
    
    def __init__(self, message, lineprop):
        self.message = message
        self.lineprop = lineprop

    def __str__(self):
        return self.message + ' in ' + str(self.lineprop)