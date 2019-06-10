# coding: utf-8

import collections
import os
import traceback

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

from chainer_compiler.ch2o.utils import new_tensor, new_sequence

from chainer_compiler.ch2o import value

def _get_trace_str():
    # TODO(hamaji): Use parsing context instead of CH2O codebase.
    skip_names = set(['_get_trace_str', 'addnode', 'calc', 'calc_seq',
                      'totensor', 'to_tensor', 'to_sequence', 'to_value_info'])
    trace = []
    for stack in reversed(traceback.extract_stack()):
        if stack.name in skip_names:
            continue
        trace.append('%s:%s:%d' %
                     (stack.name,
                      os.path.basename(stack.filename),
                      stack.lineno))
        if len(trace) == 3:
            break
    return ' '.join(trace)

class Env(object):
    def __init__(self, module):
        # Local variables keyed by their names. When a value is an
        # ONNX tensor, its value must be different from ones in other
        # variables.
        #
        # Note the type of values are `Value` or something like
        # User_Defined_*.
        #
        # TODO(hamaji): Revisit here for re-design.
        self._vars = {}
        self.nodes = []
        self.init_tensors = {}

        # Lists of a tuple of (this, key, value) where
        # - this: A `Value` object.
        # - key: A str of the attribute name.
        # - value: A `Value` object.
        self.read_attrs = []
        self.wrote_attrs = []

        self.restore_funcs = []  # User定義Linkの初期化子を正常化させるやつ
        self.module = module
        self.outer_block = None

    def get_var(self, k):
        if k in self._vars:
            return self._vars[k]

        if self.outer_block is None:
            raise NameError("name '%s' is not defined" % k)

        var = self.outer_block.get_var(k)
        if isinstance(var, value.Value):
            # Convert literals to tensors in outer scope.
            var.to_value_info(self.outer_block)
        self._vars[k] = var
        return var

    def set_var(self, k, v):
        self._vars[k] = v

    def update_vars(self, d):
        self._vars.update(d)

    def pop_var(self, k):
        return self._vars.pop(k)

    def get_var_dict(self):
        return self._vars

    def localenv(self, module):
        res = Env(module)
        res.nodes = self.nodes  # こっちはglobalに共通でないといけない
        res.init_tensors = self.init_tensors  # こっちも共通
        res.restore_funcs = self.restore_funcs
        return res

    def root(self):
        env = self
        while env.outer_block is not None:
            env = env.outer_block
        return env

    def new_block(self):
        block = Env(self.module)
        block.outer_block = self
        return block

    def addnode(self, *args, **kwargs):
        node = helper.make_node(*args, **kwargs)
        node.doc_string = _get_trace_str()
        self.nodes.append(node)

    def add_init(self, inits, pathname):
        for v in inits:
            # drint('add_init',v,p)
            v.name = pathname + v.name
            self.init_tensors[v.name] = v

    def calc(self, *args, npdtype=None, **kwargs):
        res = new_tensor(dtype=npdtype)
        assert 'outputs' not in kwargs.keys()
        kwargs['outputs'] = [res.name]
        self.addnode(*args, **kwargs)
        return res

    def calc_seq(self, *args, npdtype=None, **kwargs):
        res = new_sequence(dtype=npdtype)
        assert 'outputs' not in kwargs.keys()
        kwargs['outputs'] = [res.name]
        self.addnode(*args, **kwargs)
        return res
