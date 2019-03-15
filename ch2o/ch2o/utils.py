# coding: utf-8

import collections
import os
import traceback

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

from ch2o import value

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


_cnt = 0


def gen_id(name, prefix):
    global _cnt
    _cnt += 1
    r = prefix + str(_cnt)
    if name is not None:
        r = name + '_' + r
    return r


def new_tensor(dims=None, dtype=None, name=None):
    if dtype is not None:
        dt = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
    else:
        # TODO(hamaji): Deprecate this fallback pass.
        dt = onnx.TensorProto.FLOAT
    return helper.make_tensor_value_info(gen_id(name, 'T'), dt, dims)


def new_sequence(dtype=None, name=None):
    if dtype is not None:
        dt = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    else:
        # TODO(hamaji): Deprecate this fallback pass.
        dt = onnx.TensorProto.FLOAT
    vi = onnx.ValueInfoProto()
    vi.name = gen_id(name, 'S')
    vi.type.sequence_type.elem_type.tensor_type.elem_type = dt
    return vi


_graph_ids = {}


def gen_graph_name(name):
    if name in _graph_ids:
        _graph_ids[name] += 1
    else:
        _graph_ids[name] = 0
    return '%s_%d' % (name, _graph_ids[name])


def get_dims(tensor):
    dims = tensor.type.tensor_type.shape.dim
    return list(map(lambda x: x.dim_value, dims))


def clip_head(s):
    s = s.split('\n')
    # print(s)
    hs = os.path.commonprefix(list(filter(lambda x: x != '', s)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    s = map(lambda x: x[ls:], s)
    return '\n'.join(s)


class ValueReturn(Exception):
    def __init__(self, value):
        self.value = value


def size2d(x):
    if isinstance(x, collections.Iterable):
        return x
    return x, x


def istensor(x):
    return isinstance(x, onnx.ValueInfoProto)


def totensor(x, env, dtype=None):
    if istensor(x):
        assert dtype is None
        return x

    # We use a scalar false as a None.
    # TODO(hamaji): Revisit to check if this decision is OK.
    if x is None:
        x = False

    if type(x) == tuple or type(x) == list:
        def f(v):
            tv = v.to_tensor(env)
            tw = env.calc(
                'Unsqueeze',
                inputs=[tv.name],
                axes=[0]
            )
            return tw.name

        vs = list(map(f, x))
        # print(vs)
        res = env.calc(
            'Concat',
            inputs=vs,
            axis=0
        )
    else:
        if dtype is None and type(x) == float:
            dtype = np.float32
        x = np.array(x, dtype=dtype)
        dt = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[x.dtype]
        res = env.calc(
            'Constant',
            inputs=[],
            value=onnx.helper.make_tensor(
                name="hoge",
                data_type=dt,
                dims=x.shape,
                vals=x.flat,
            )
        )

    return res


def onnx_dtype(dtype):
    a = np.zeros((), dtype=dtype)
    dt = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[a.dtype]
    return dt


def make_graph(nodes, graph_name, inputs, outputs):
    input_dict = {}
    for input in inputs:
        input_dict[input.name] = (input, None)
    outputs_fixed = []
    for output in outputs:
        if output.name in input_dict:
            input, new_output = input_dict[output.name]
            if new_output is None:
                new_output = new_tensor(name=graph_name + '_out')
                nodes.append(helper.make_node('Identity',
                                              inputs=[input.name],
                                              outputs=[new_output.name]))
                input_dict[output.name] = (input, new_output)
        else:
            new_output = output
        outputs_fixed.append(new_output)

    graph_name = gen_graph_name(graph_name)
    return helper.make_graph(nodes, graph_name, inputs, outputs_fixed)
