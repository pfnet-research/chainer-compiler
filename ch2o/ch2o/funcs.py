# coding: utf-8

import collections
import onnx
from onnx import helper
from onnx import TensorProto

import chainer
from chainer import functions as F
import numpy as np

from ch2o import utils
from ch2o.env import Env
from ch2o.utils import new_tensor, get_dims, size2d, istensor, totensor, clip_head
from ch2o.callable import Callable
from ch2o.value import Value

import ast
import code
import gast

from typing import List, Mapping


class Function_SimpleUnary(Callable):
    def __init__(self, fn, onnx_name):
        super(Function_SimpleUnary, self).__init__(fn)
        self.onnx_name = onnx_name

    def call_impl(self, env, v):
        return env.calc(self.onnx_name, inputs=[v.to_tensor(env).name])


def _int_or_list(v: Value) -> List[int]:
    if not v.is_py:
        raise TypeError('Expected an int or an int list: %s' % v.value)
    if isinstance(v.value, collections.Iterable):
        assert all(x.is_py for x in v.value)
        return list(x.value for x in v.value)
    return [v.value]


def _pair(v: Value) -> List[int]:
    if not v.is_py:
        raise TypeError('Expected an int or an int list: %s' % v.value)
    if isinstance(v.value, collections.Iterable):
        assert all(x.is_py for x in v.value)
        return list(x.value for x in v.value)
    return [v.value, v.value]


class Function_MaxPool2d(Callable):
    def call_impl(self, env, x, ksize, stride, pad, cover_all, return_indices):
        assert not return_indices.value  # TODO(hamaji): Not implemented yet.
        kwargs = {}
        if not stride.is_none():
            kwargs['strides'] = _pair(stride)
        if pad.value:
            kwargs['pads'] = _pair(pad) * 2
        return env.calc(
            'MaxPool',
            inputs=[x.to_tensor(env).name],
            kernel_shape=_pair(ksize),
            onikux_cover_all=cover_all.to_bool(),
            **kwargs)


class Function_AveragePool2d(Callable):
    def call_impl(self, env, x, ksize, stride, pad):
        kwargs = {}
        if not stride.is_none():
            kwargs['strides'] = _pair(stride)
        if pad.value:
            kwargs['pads'] = _pair(pad) * 2
        return env.calc(
            'AveragePool',
            inputs=[x.to_tensor(env).name],
            kernel_shape=_pair(ksize),
            **kwargs)


class Function_LocalRespNorm(Callable):
    def call_impl(self, env, x, n, k, alpha, beta):
        n = n.to_int()
        return env.calc(
            "LRN",
            inputs=[x.to_tensor(env).name],
            size=n,
            bias=k.to_float(),
            alpha=alpha.to_float() * n,  # chainerとonnx(mxnet)で一致しない
            beta=beta.to_float()
        )


class Function_Dropout(Callable):
    def call_impl(self, env, x, ratio, **kwargs):
        assert not kwargs  # TODO(hamaji): Not supported yet.
        return env.calc("Dropout",
                        inputs=[x.to_tensor(env).name], ratio=ratio.to_float())


class Function_Matmul(Callable):
    def call_impl(self, env, a, b, transa, transb):
        assert not transa.value  # TODO(hamaji): Not supported yet.
        assert not transb.value  # TODO(hamaji): Not supported yet.
        return env.calc("MatMul",
                        inputs=[a.to_tensor(env).name,
                                b.to_tensor(env).name])


class Function_Concat(Callable):
    def call_impl(self, env, xs, axis):
        if isinstance(xs.value, tuple):
            return env.calc(
                "Concat",
                inputs=list(x.to_tensor(env).name for x in xs.value),
                axis=axis.to_int(),
            )
        else:
            return env.calc(
                "OnikuxSequenceConcat",
                inputs=[xs.to_sequence(env).name],
                axis=axis.to_int(),
            )


class Function_SoftmaxCrossEntropy(Callable):
    def call_impl(self, env, x, t, normalize, cache_score, class_weight, ignore_label, reduce, enable_double_backprop):
        assert normalize.value  # TODO(hamaji): Not supported yet.
        assert cache_score.value  # TODO(hamaji): Not supported yet.
        assert class_weight.value is None  # TODO(hamaji): Not supported yet.
        assert ignore_label.value == -1  # TODO(hamaji): Not supported yet.
        assert reduce.value == 'mean'  # TODO(hamaji): Not supported yet.
        assert not enable_double_backprop.value  # TODO(hamaji): Not supported yet.
        return env.calc(
            "OnikuxSoftmaxCrossEntropy",
            inputs=[x.to_tensor(env).name, t.to_tensor(env).name],
        )


class Function_PadSequence(Callable):
    def call_impl(self, env, xs, length, padding):
        kwargs = {}
        if not length.is_none():
            kwargs['length'] = length.to_int()
        if not padding.is_none():
            kwargs['value'] = padding.to_float()
        return env.calc(
            "OnikuxSequencePad",
            inputs=[xs.to_sequence(env).name],
            **kwargs
        )


class Function_SwapAxes(Callable):
    def call_impl(self, env, x, axis1, axis2):
        a = axis1.to_int()
        b = axis2.to_int()
        pe = list(range(max(a, b)+1))
        pe[a] = b
        pe[b] = a

        return env.calc(
            "Transpose",
            inputs=[x.to_tensor(env).name],
            perm=pe
        )


class Function_Reshape(Callable):
    def call_impl(self, env, x, shape):
        return env.calc(
            "Reshape",
            inputs=[x.to_tensor(env).name, shape.to_tensor(env).name]
        )


class Function_ExpandDims(Callable):
    def call_impl(self, env, x, axis):
        return env.calc(
            "Unsqueeze",
            inputs=[x.to_tensor(env).name],
            axes=[axis.to_int()],
        )


class Function_BroadcastTo(Callable):
    def call_impl(self, env, x, shape):
        return env.calc(
            "Expand",
            inputs=[x.to_tensor(env).name, shape.to_tensor(env).name]
        )


def castto(v, tt, env):
    res = env.calc(
        'Cast',
        inputs=[v.name],
        to=tt
    )
    res.type.tensor_type.elem_type = tt
    return res


class Np_Array(Callable):
    def __init__(self, _):
        def a(object, dtype=None, copy=True,
              order='K', subok=False, ndmin=0):
            pass
        super(Np_Array, self).__init__(a)

    def call_impl(self, env, object, dtype, copy, order, subok, ndmin):
        assert copy.value is True  # TODO(hamaji): Not supported yet.
        assert order.value == 'K'  # TODO(hamaji): Not supported yet.
        assert subok.value is False   # TODO(hamaji): Not supported yet.
        assert ndmin.value == 0  # TODO(hamaji): Not supported yet.
        return object.to_tensor(env, dtype=dtype.value)


class Np_Int32(Callable):
    def __init__(self, _):
        super(Np_Int32, self).__init__(lambda x: x)

    def call(self, env, x):
        return castto(x.to_tensor(env).name, TensorProto.INT32, env)


class Np_Zeros(Callable):
    def __init__(self, _):
        def a(shape, dtype=float, order='C'):
            pass
        super(Np_Zeros, self).__init__(a)

    def call_impl(self, env, shape, dtype, order):
        assert order.value == 'C'
        dt = utils.onnx_dtype(dtype.value)
        return env.calc(
            'ConstantFill',
            inputs=[shape.to_tensor(env).name],
            input_as_shape=1,
            dtype=dt
        )


class Np_Full(Callable):
    def call_impl(self, env, shape, fill_value, dtype, order):
        assert order.value == 'C'
        res = env.calc(
            'Expand',
            inputs=[fill_value.to_tensor(env).name,
                    shape.to_tensor(env).name],
        )
        if not dtype.is_none():
            dt = utils.onnx_dtype(dtype.value)
            res = castto(res, dt, env)
        return res


class Np_Cumsum(Callable):
    def __init__(self, _):
        super(Np_Cumsum, self).__init__(
            lambda a, axis=None, dtype=None, out=None: a)

    def call_impl(self, env, a, axis, dtype, out):
        assert axis.is_none()  # TODO(hamaji): Not supported yet.
        assert dtype.is_none()  # TODO(hamaji): Not supported yet.
        assert out.is_none()  # TODO(hamaji): Not supported yet.
        # さらにさらに、入力は1次元のTensorである、と仮定してしまいます
        # 戻り値は入力に依らずテンソルらしい
        # TODO(satos) さすがに仮定がきつい
        v = a.to_tensor(env)

        # これ戻り値がテンソルでなくSequenceなら、SplitAxisみたいにかっこよく書けるはず
        """
        a = new_tensor()
        env.addnode(
            'Flatten',
            inputs=[v.name],outputs[a.name],
            axis=0
        )
        v = a
        a = new_tensor()
        env.addnode(
            'Squeeze',
            inputs=[v.name],outputs[a.name],
            axes=[0]
        )
        """
        ls = env.calc(
            'OnikuxGenericLen',
            inputs=[v.name],
        )

        def dummy():
            return "dummy_" + new_tensor().name

        localenv = Env(env.module)
        cnt = new_tensor()
        cond = new_tensor()
        s = new_tensor()
        gtx = new_tensor()
        tx = localenv.calc(
            "OnikuxGenericGetItem",
            inputs=[gtx.name, cnt.name],
        )
        ts = localenv.calc(
            "Add",
            inputs=[tx.name, s.name],
        )

        zero = totensor(0, env)

        res = new_tensor()
        env.addnode(
            'Loop',
            inputs=[ls.name, "", v.name, zero.name],
            outputs=[dummy(), dummy(), res.name],
            body=helper.make_graph(
                localenv.nodes,
                utils.gen_graph_name("Cumsum_subgraph"),
                [cnt, cond, gtx, s],
                [cond, gtx, ts, ts]
            )
        )

        return res


class Function_SplitAxis(Callable):
    def call_impl(self, env, x, indices_or_sections, axis, force_tuple):
        assert axis.value == 0
        assert force_tuple.value is True
        # さらにさらに、入力は1次元のTensorである、と仮定してしまいます
        # 戻り値はtuple(!!)らしいが、たってきSequenceで返してます。
        # TODO(satos) さすがに仮定がきつい

        v = x
        ilens = indices_or_sections

        from . chainer2onnx import eval_ast

        src = """
        r = []
        bs = 0
        for s in ilens:
            r.append(v[bs:s])
            bs = s
        r.append(v[bs:])
        """
        src = clip_head(src)
        nast = gast.ast_to_gast(ast.parse(src))

        localenv = Env({})
        vs = {
            'v': v,
            'ilens': ilens,
        }
        localenv.update_vars(vs)
        eval_ast(nast.body, localenv)

        env.nodes += localenv.nodes
        return localenv.get_var('r')


class Xp_Np_Ceil(Callable):
    def __init__(self, _):
        super(Xp_Np_Ceil, self).__init__(lambda x: x)

    def call_impl(self, env, x):
        return env.calc('Ceil', inputs=[x.to_tensor(env).name])


class Cuda_ToCpu(object):
    def call(self, args, _, env):
        assert len(args) == 1
        # TODO(satos) gpuからcpuに移ったというデータをどうにかして載せる
        return args[0]


class Function_Vstack(Callable):
    def call_impl(self, env, xs):
        return env.calc(
            'OnikuxSequenceConcat',
            inputs=[xs.to_sequence(env).name],
            axis=0
        )


class Function_Hstack(Callable):
    def call_impl(self, env, xs):
        return env.calc(
            'OnikuxSequenceConcat',
            inputs=[xs.to_sequence(env).name],
            axis=1
        )


class Function_Stack(Callable):
    def call_impl(self, env, xs, axis):
        return env.calc(
            'OnikuxSequenceStack',
            inputs=[xs.to_sequence(env).name],
            axis=axis.to_int()
        )


class Function_Separate(Callable):
    def call_impl(self, env, x, axis):
        return env.calc_seq(
            'OnikuxSequenceSeparate',
            inputs=[x.to_tensor(env).name],
            axis=axis.to_int()
        )


class Function_Squeeze(Callable):
    def call_impl(self, env, x, axis):
        assert not axis.is_none()  # TODO(hamaji): Not supported yet.
        return env.calc(
            'Squeeze',
            inputs=[x.to_tensor(env).name],
            axes=_int_or_list(axis)
        )


class Function_Sum(Callable):
    def call_impl(self, env, x, axis, keepdims):
        if not axis.has_py_value:
            raise TypeError('Expected an int or an int tuple: %s' % axis.value)
        axis = axis.to_py_value()
        if isinstance(axis.value, collections.Iterable):
            axes = axis.to_int_list()
        elif axis.is_none():
            axes = []
        else:
            axes = [axis.to_int()]
        kwargs = {}
        if axes:
            kwargs['axes'] = axes
        return env.calc(
            'ReduceSum',
            inputs=[x.to_tensor(env).name],
            keepdims=keepdims.to_bool(),
            **kwargs
        )


class Function_Average(Callable):
    def call_impl(self, env, x, axis, weights, keepdims):
        assert weights.is_none()  # TODO(hamaji): Not supported yet.
        if not axis.has_py_value:
            raise TypeError('Expected an int or an int tuple: %s' % axis.value)
        axis = axis.to_py_value()
        if isinstance(axis.value, collections.Iterable):
            axes = axis.to_int_list()
        elif axis.is_none():
            axes = []
        else:
            axes = [axis.to_int()]
        kwargs = {}
        if axes:
            kwargs['axes'] = axes
        return env.calc(
            'ReduceMean',
            inputs=[x.to_tensor(env).name],
            keepdims=keepdims.to_bool(),
            **kwargs
        )


class Function_Softmax(Callable):
    def call_impl(self, env, x, axis):
        return env.calc(
            'Softmax',
            inputs=[x.to_tensor(env).name],
            axis=axis.to_int()
        )


class Function_Chainer_Variable(Callable):
    def call_impl(self, env, data, **kwargs):
        assert not kwargs   # TODO(hamaji): Not supported yet.
        return env.calc(
            'Identity',
            inputs=[data.to_value_info(env).name],
        )


class Function_Dummy(object):
    def __init__(self, s=""):
        self.name = s

    def call(self, args, keywords, env):
        # raise Exception(self,"Unimplemented")
        return env.calc(
            "Dummy of %s and should be removed" % self.name,
            inputs=[]
        )


dummies = [
    F.flatten,
    F.accuracy,
]


class Func(object):
    def __init__(self, f):
        self.call = f


Func2NodeClass = dict([
    (chainer.backends.cuda.to_cpu, Cuda_ToCpu()),

    # TODO(satos) とりあえずhttps://github.com/espnet/espnet/blob/master/src/nets/deterministic_embe    d_id.py#L43) のif文を通らないようにする
    (chainer.utils.type_check.same_types, Func(lambda _, __, ___: True)),
] + (
    list(map(lambda f: (f, Function_Dummy(f)), dummies))
))

for fn, cls in [(F.expand_dims, Function_ExpandDims),
                (F.broadcast_to, Function_BroadcastTo),
                (F.reshape, Function_Reshape),
                (F.dropout, Function_Dropout),
                (F.matmul, Function_Matmul),
                (F.max_pooling_2d, Function_MaxPool2d),
                (F.average_pooling_2d, Function_AveragePool2d),
                (F.local_response_normalization, Function_LocalRespNorm),
                (F.concat, Function_Concat),
                (F.softmax_cross_entropy, Function_SoftmaxCrossEntropy),
                (F.pad_sequence, Function_PadSequence),
                (F.swapaxes, Function_SwapAxes),
                (F.split_axis, Function_SplitAxis),
                (np.array, Np_Array),
                (np.int32, Np_Int32),
                (np.ceil, Xp_Np_Ceil),
                (np.zeros, Np_Zeros),
                (np.full, Np_Full),
                (np.cumsum, Np_Cumsum),
                (F.vstack, Function_Vstack),
                (F.hstack, Function_Hstack),
                (F.stack, Function_Stack),
                (F.separate, Function_Separate),
                (F.sum, Function_Sum),
                (F.average, Function_Average),
                (F.softmax, Function_Softmax),
                (F.squeeze, Function_Squeeze),
                (chainer.Variable, Function_Chainer_Variable),
]:
    Func2NodeClass[fn] = cls(fn)

for fn, name in [(F.relu, 'Relu'),
                 (F.sigmoid, 'Sigmoid'),
                 (F.tanh, 'Tanh')]:
    Func2NodeClass[fn] = Function_SimpleUnary(fn, name)
