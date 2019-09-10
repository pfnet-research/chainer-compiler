import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer.utils.conv import get_conv_outsize
from chainer.utils import size_of_shape

from chainer_compiler.elichika.typing.types import *


def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    return x


def get_kwarg(ty_kwargs, key, default=None):
    if key in ty_kwargs.keys():
        # TODO(momohatt): when unable to get the correct value, do something
        return value_of_type(ty_kwargs[key]), lacks_value(ty_kwargs[key])
    return default, False


def assert_dtype_equal(expected, actual):
    msg = "dtype mismatch (Expected: {}, got: {})".format(expected, actual)
    assert expected == actual, msg


def marunage(func, args, ty_kwargs, is_fake_shape, is_fake_dtype):
    dummy_kwargs = {k : value_of_type(t) for (k, t) in ty_kwargs.items()}
    dummy_result = func(*args, **dummy_kwargs)
    ty_result = type_of_value(dummy_result)

    if isinstance(ty_result, TyTensor):
        if is_fake_shape: ty_result.shape = None
        if is_fake_dtype: ty_result.dtype = None
    elif isinstance(ty_result, TySequence):
        assert ty_result.is_fixed_len
        if is_fake_shape:
            for t in ty_result.get_tys():
                t.shape = None
        if is_fake_dtype:
            for  t in ty_result.get_tys():
                t.dtype = None

    return ty_result


def make_infer(func, fallback_shapes, fallback_dtypes):
    def infer(ty_args, ty_kwargs):
        ty_args_tensor = [t for t in ty_args if isinstance(t, TyTensor)]

        shapes = [s if t.shape is None else t.shape
                for t, s in zip(ty_args_tensor, fallback_shapes)]
        dtypes = [dt if t.dtype is None else t.dtype
                for t, dt in zip(ty_args_tensor, fallback_dtypes)]
        is_dummy_shape = any([t.shape is None for t in ty_args_tensor])
        is_dummy_dtype = any([t.dtype is None for t in ty_args_tensor])

        # XXX: tensor arguments always come before non-tensor arguments
        dummy_args = [np.zeros(s, t) for s, t in zip(shapes, dtypes)] + \
                [value_of_type(t) for t in ty_args if not isinstance(t, TyTensor)]
        dummy_kwargs = {k : value_of_type(t) for (k, t) in ty_kwargs.items()}
        dummy_result = func(*dummy_args, **dummy_kwargs)
        ty_result = type_of_value(dummy_result)
        if isinstance(ty_result, TyTensor):
            if is_dummy_shape:
                ty_result.shape = None
            if is_dummy_dtype:
                ty_result.dtype = None
        return ty_result

    return infer


def ty_NumpyArray(ty_args, ty_kwargs):
    infer = make_infer(np.array, (), ())
    return infer(ty_args, ty_kwargs)


def ty_NumpyOnes(ty_args, ty_kwargs):
    shape_type = ty_args[0]
    is_dummy_shape = lacks_value(shape_type)
    shape = value_of_type(shape_type)
    dtype, is_dummy_dtype = get_kwarg(ty_kwargs, 'dtype', np.dtype('float64'))

    ty_ret = TyNdarray(dtype=dtype, shape=shape)
    if is_dummy_shape: ty_ret.shape = None
    if is_dummy_dtype: ty_ret.dtype = None
    return ty_ret


def ty_NumpyFull(ty_args, ty_kwargs):
    shape_type = ty_args[0]
    value_type = ty_args[1]
    dtype, is_dummy_dtype = get_kwarg(ty_kwargs, 'dtype', tyobj2dtype(value_type))

    assert isinstance(shape_type, TyNum) or isinstance(shape_type, TyTuple)
    is_dummy_shape = lacks_value(shape_type)
    shape = value_of_type(shape_type)
    if isinstance(shape, int):
        shape = (shape, )

    y_type = TyNdarray(dtype=dtype, shape=shape)
    if is_dummy_dtype: y_type.dtype = None
    if is_dummy_shape: y_type.shape = None
    return y_type


def ty_ChainerVariable(ty_args, ty_kwargs):
    infer = make_infer(chainer.Variable, (1,), (np.float32,))
    return infer(ty_args, ty_kwargs)


class ty_ChainerPooling2d():
    # max_pooling_2d / average_pooling_2d
    def __init__(self, func):
        self.func = func

    def __call__(self, ty_args, ty_kwargs):
        # TODO(momohatt): handle cases where ksize is unknown
        ksize = ty_args[1].value
        # TODO(momohatt): use is_dummy_stride
        stride, is_dummy_stride = get_kwarg(ty_kwargs, 'stride', default=ksize)
        minimum_size = max(ksize, stride)
        fallback_shapes = ((1, 1, minimum_size, minimum_size),)
        fallback_dtypes = (np.float32,)

        infer = make_infer(self.func, fallback_shapes, fallback_dtypes)
        return infer(ty_args, ty_kwargs)


def ty_ChainerSoftmaxCrossEntropy(ty_args, ty_kwargs):
    x_shape, t_shape = ty_args[0].shape, ty_args[1].shape
    fallback_dtypes = (np.float32, np.int64)

    # x.shape[0] == t.shape[0]
    if x_shape is None and t_shape is not None:
        fallback_shapes = ((t_shape[0], 1), t_shape)
    elif x_shape is not None and t_shape is None:
        fallback_shapes = (x_shape, (x_shape[0],))
    else:
        fallback_shapes = ((1, 1), (1,))

    infer = make_infer(
            F.softmax_cross_entropy, fallback_shapes, fallback_dtypes)
    return infer(ty_args, ty_kwargs)


class ty_ChainerIdentical():
    # functions that doesn't change shapes or dtypes

    def __init__(self, is_float_only=True):
        self.is_float_only = is_float_only

    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        if self.is_float_only:
            assert_dtype_equal('f', x_type.dtype.kind)
        return copy_ty(x_type)


class ty_ChainerConcat():
    # concat, hstack, vstack, stack

    def __init__(self, func):
        self.func = func

    def __call__(self, ty_args, ty_kwargs):
        xs_type = ty_args[0]

        assert isinstance(xs_type, TySequence)
        if not xs_type.is_fixed_len:
            dtype = xs_type.get_ty().dtype
            return TyChainerVariable(dtype=dtype)

        xs_dtypes = [x_type.dtype for x_type in xs_type.get_tys()]
        assert all_same(xs_dtypes)

        axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', default=1)

        if lacks_value(xs_type) or is_dummy_axis:
            return TyChainerVariable(dtype=xs_dtypes[0])

        xs = value_of_type(xs_type)

        if self.func is F.vstack or self.func is F.hstack:
            return type_of_value(self.func(xs))

        return type_of_value(self.func(xs, axis=axis))


def ty_ChainerExpandDims(ty_args, ty_kwargs):
    axis = ty_args[1].value
    fallback_shapes = ((1,) * axis,)
    fallback_dtypes = (np.float32,)

    infer = make_infer(F.expand_dims, fallback_shapes, fallback_dtypes)
    return infer(ty_args, ty_kwargs)


def ty_ChainerBroadcastTo(ty_args, ty_kwargs):
    x_type = ty_args[0]
    shape_type = ty_args[1]

    if x_type.shape is None or lacks_value(shape_type):
        return TyChainerVariable(dtype=x_type.dtype)

    in_shape = x_type.shape
    out_shape = value_of_type(shape_type)

    # check_type_forward
    ndim = len(out_shape)
    assert len(in_shape) <= ndim

    for i in range(-1, - len(in_shape) - 1, -1):
        assert in_shape[i] == out_shape[i] or in_shape[i] == 1

    return TyChainerVariable(dtype=x_type.dtype, shape=out_shape)


def ty_ChainerSum(ty_args, ty_kwargs):
    # TODO(momohatt): use is_dummy_axis
    axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', default=None)
    fallback_shapes = ((1,) * (axis + 1),)
    fallback_dtypes = (np.float32,)

    infer = make_infer(F.sum, fallback_shapes, fallback_dtypes)
    return infer(ty_args, ty_kwargs)


def infer_reshape(orig_shape, input_shape):
    # orig_shape can be None
    # input_shape should be accurate
    if orig_shape is None:
        if any([i == -1 for i in input_shape]):
            return None
        return input_shape
    fill = abs(size_of_shape(orig_shape) // size_of_shape(input_shape))
    ret_shape = tuple([i if i != -1 else fill for i in input_shape])
    assert size_of_shape(orig_shape) == size_of_shape(ret_shape)
    return ret_shape


def ty_ChainerReshape(ty_args, ty_kwargs):
    x_type = ty_args[0]
    shape_type = ty_args[1]

    if lacks_value(shape_type):
        return TyChainerVariable(dtype=x_type.dtype, shape=None)
    ret_shape = infer_reshape(x_type.shape, value_of_type(shape_type))
    return TyChainerVariable(dtype=x_type.dtype, shape=ret_shape)


def ty_ChainerSqueeze(ty_args, ty_kwargs):
    x_type = ty_args[0]

    # TODO: don't use F.squeeze
    axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', None)

    if x_type.shape is None or lacks_value(x_type) or is_dummy_axis:
        return TyChainerVariable(dtype=x_type.dtype)

    ret = F.squeeze(value_of_type(x_type), axis=axis)
    return type_of_value(ret)


def ty_ChainerSwapAxes(ty_args, ty_kwargs):
    def swap(t, i, j):
        l = list(t)
        l[i], l[j] = l[j], l[i]
        return tuple(l)

    x_type = ty_args[0]
    axis1_type = ty_args[1]
    axis2_type = ty_args[2]

    if x_type.shape is None or \
            lacks_value(axis1_type) or lacks_value(axis2_type):
        return TyChainerVariable(dtype=x_type.dtype)

    shape = x_type.shape
    axis1 = value_of_type(axis1_type)
    axis2 = value_of_type(axis2_type)

    assert axis1 < len(shape) and axis2 < len(shape)

    return TyChainerVariable(dtype=x_type.dtype,
            shape=swap(shape, axis1, axis2))


def ty_ChainerSeparate(ty_args, ty_kwargs):
    x_type = ty_args[0]
    axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', 0)
    assert isinstance(x_type, TyTensor)
    x_shape = x_type.shape
    x_dtype = x_type.dtype

    if x_shape is None or is_dummy_axis:
        return TyTuple(TyChainerVariable(dtype=x_dtype))

    assert axis < len(x_shape)

    n = x_shape[axis]
    out_shape = x_shape[:axis] + x_shape[axis + 1:]
    return TyTuple([TyChainerVariable(dtype=x_dtype, shape=out_shape)] * n)


def ty_ChainerSplitAxis(ty_args, ty_kwargs):
    x_type = ty_args[0]

    assert isinstance(x_type, TyTensor)

    if isinstance(ty_args[1], TyNum):
        n = ty_args[1].value
        if n is None:
            # TODO
            return TyVar()
        return TyTuple([TyChainerVariable(dtype=x_type.dtype)] * n)

    assert isinstance(ty_args[1], TyTensor)
    # 1-D array

    if ty_args[1].shape is None:
        # variable length tuple
        return TyTuple(TyChainerVariable(dtype=x_type.dtype))

    assert len(ty_args[1].shape) == 1
    n = ty_args[1].shape[0]
    return TyTuple([TyChainerVariable(dtype=x_type.dtype)] * n)


def ty_ChainerPadSequence(ty_args, ty_kwargs):
    # shape[1:] should be uniform & ndim > 0
    xs_type = ty_args[0]
    assert isinstance(xs_type, TySequence)
    is_dummy_shape = False
    if xs_type.is_fixed_len:
        is_shape_None = [t.shape is None for t in xs_type.get_tys()]
        if any(is_shape_None):
            is_dummy_shape = True
            if all(is_shape_None):
                dummy_arg = [np.zeros((1,), dtype=t.dtype) for t in xs_type.get_tys()]
            else:
                t = utils.find(xs_type.get_tys(), lambda t: t.shape is not None)
                fallback_shape = t.shape
                dummy_arg = [
                        np.zeros(fallback_shape, dtype=t.dtype) if t.shape is None
                        else np.zeros(t.shape, dtype=t.dtype) for t in xs_type.get_tys()]
        else:
            dummy_arg = value_of_type(xs_type)

    else:
        is_dummy_shape = True
        dummy_arg = [np.zeros((1,), dtype=xs_type.get_ty().dtype)]

    dummy_kwargs = {k : value_of_type(t) for (k, t) in ty_kwargs.items()}
    ty_ret = type_of_value(F.pad_sequence(dummy_arg, **dummy_kwargs))
    if is_dummy_shape:
        ty_ret.shape = None
    return ty_ret


def ty_ChainerLocalResponseNormalization(ty_args, ty_kwargs):
    infer = make_infer(F.local_response_normalization,
            (1, 1), (np.float32,))
    return infer(ty_args, ty_kwargs)

# ================================= Links ======================================

class ty_ChainerLinear():
    def __call__(self, linear, ty_args, ty_kwargs):
        x_type = ty_args[0]
        n_batch_axes, is_dummy_n_batch_axes = \
                get_kwarg(ty_kwargs, 'n_batch_axes', default=1)

        if x_type.dtype is not None and linear.b is not None:
            assert_dtype_equal(linear.b.dtype, x_type.dtype)
        if x_type.shape is None or is_dummy_n_batch_axes:
            return TyChainerVariable(dtype=x_type.dtype, shape=None)

        out_shape = self.infer_return_shape(linear, x_type.shape, n_batch_axes)
        return TyChainerVariable(dtype=x_type.dtype, shape=out_shape)

    def infer_return_shape(self, linear, x_shape, n_batch_axes):
        assert n_batch_axes >= 1

        if n_batch_axes > 1:
            batch_shape = x_shape[:n_batch_axes]
            batch_size = size_of_shape(batch_shape)
            x_shape = infer_reshape(x_shape, (batch_size, -1))
        elif len(x_shape) > 2:
            x_shape = infer_reshape(x_shape, (x_shape[0], -1))

        if linear.in_size is not None:
            assert x_shape[1] == linear.in_size

        y_shape = (x_shape[0], linear.out_size)
        if n_batch_axes > 1:
            y_shape = infer_reshape(y_shape, (batch_shape + (-1,)))
        return y_shape


class ty_ChainerConvolution2D():
    def __call__(self, conv, ty_args, ty_kwargs):
        x_shape = ty_args[0].shape
        x_dtype = ty_args[0].dtype
        if x_dtype is not None:
            assert_dtype_equal('f', x_dtype.kind)
        if x_shape is None:
            return TyChainerVariable(dtype=x_dtype, shape=None)

        assert len(x_shape) == 4
        if conv.in_channels is not None:
            assert x_shape[1] == conv.in_channels

        y_shape = self.infer_return_shape(conv, x_shape)
        return TyChainerVariable(dtype=x_dtype, shape=y_shape)

    def infer_return_shape(self, conv, x_shape):
        ksize = make_pair(conv.ksize)
        stride = make_pair(conv.stride)
        pad = make_pair(conv.pad)
        dilate = make_pair(conv.dilate)

        shape_2 = get_conv_outsize(
                x_shape[2], ksize[0], stride[0], pad[0], d=dilate[0])
        shape_3 = get_conv_outsize(
                x_shape[3], ksize[1], stride[1], pad[1], d=dilate[1])
        return (x_shape[0], conv.out_channels, shape_2, shape_3)


def ty_ChainerBatchNormalization(obj, ty_args, ty_kwargs):
    assert False


def ty_ChainerEmbedID(embed, ty_args, ty_kwargs):
    assert isinstance(ty_args[0], TyTensor)
    x_type = ty_args[0]
    w_type = embed.W

    if x_type.shape is None:
        return TyChainerVariable(dtype=w_type.dtype)

    assert_dtype_equal('i', x_type.dtype.kind)
    assert len(x_type.shape) >= 1

    assert all([t < w_type.shape[0] for t in x_type.shape])
    out_type = x_type.shape + (w_type.shape[1],)
    return TyChainerVariable(dtype=w_type.dtype, shape=out_shape)


def ty_ChainerNStepBiLSTM(nblstm, ty_args, ty_kwargs):
    hx_type = ty_args[0]
    cx_type = ty_args[1]
    xs_type = ty_args[2]
    assert isinstance(xs_type, TySequence)

    if not xs_type.is_fixed_len:
        # TODO
        return TyTuple([hx_type, cx_type, xs_type])

    xs_dtypes = [t.dtype for t in xs_type.get_tys()]
    xs_shapes = [t.shape for t in xs_type.get_tys()]
    assert all_same(xs_dtypes)

    if isinstance(hx_type, TyTensor):
        hx_shape = hx_type.shape
        hx_dtype = hx_type.dtype
    else:
        hx_shape = (nblstm.n_layers * 2, len(xs_type.get_tys()), nblstm.out_size)
        hx_dtype = xs_type.get_tys()[0].dtype

    if isinstance(cx_type, TyTensor):
        cx_shape = cx_type.shape
        cx_dtype = cx_type.dtype
    else:
        cx_shape = (nblstm.n_layers * 2, len(xs_type.get_tys()), nblstm.out_size)
        cx_dtype = hx_dtype

    if hx_shape is None or cx_shape is None:
        # TODO: 適当
        return TyTuple([hx_type, cx_type, xs_type])

    assert hx_shape[0] // 2 == nblstm.n_layers
    assert hx_shape == cx_shape
    N = hx_shape[2]

    hy_type = TyChainerVariable(dtype=hx_dtype, shape=hx_shape)
    cy_type = TyChainerVariable(dtype=cx_dtype, shape=cx_shape)

    if any([t.shape is None for t in xs_type.get_tys()]):
        return TyTuple([hy_type, cy_type, TyList(TyChainerVariable(dtype=xs_dtypes[0]))])

    # TODO(momohatt): nblstm doesn't have attribute in_size
    # assert all([i == nblstm.in_size for _, i in xs_shapes])
    ys_shapes = [(l, 2 * N) for l, _ in xs_shapes]
    ys_type = TyList([TyChainerVariable(dtype=xs_dtypes[0], shape=s) for s in ys_shapes])

    return TyTuple([hy_type, cy_type, ys_type])


ext_func_ty = {
        np.array                       : ty_NumpyArray,
        np.cumsum                      : ty_ChainerIdentical(is_float_only=False),
        np.full                        : ty_NumpyFull,
        np.ones                        : ty_NumpyOnes,
        np.zeros                       : ty_NumpyOnes,
        chainer.Variable               : ty_ChainerVariable,
        cuda.to_cpu                    : ty_ChainerIdentical(is_float_only=False),
        F.average_pooling_2d           : ty_ChainerPooling2d(F.average_pooling_2d),
        F.broadcast_to                 : ty_ChainerBroadcastTo,
        F.concat                       : ty_ChainerConcat(F.concat),
        F.dropout                      : ty_ChainerIdentical(),
        F.expand_dims                  : ty_ChainerExpandDims,
        F.hstack                       : ty_ChainerConcat(F.hstack),
        F.local_response_normalization : ty_ChainerLocalResponseNormalization,
        F.max_pooling_2d               : ty_ChainerPooling2d(F.max_pooling_2d),
        F.pad_sequence                 : ty_ChainerPadSequence,
        F.relu                         : ty_ChainerIdentical(),
        F.reshape                      : ty_ChainerReshape,
        F.separate                     : ty_ChainerSeparate,
        F.sigmoid                      : ty_ChainerIdentical(),
        F.split_axis                   : ty_ChainerSplitAxis,
        F.squeeze                      : ty_ChainerSqueeze,
        F.softmax                      : ty_ChainerIdentical(),
        F.softmax_cross_entropy        : ty_ChainerSoftmaxCrossEntropy,
        F.stack                        : ty_ChainerConcat(F.stack),
        F.sum                          : ty_ChainerSum,
        F.swapaxes                     : ty_ChainerSwapAxes,
        F.tanh                         : ty_ChainerIdentical(),
        F.vstack                       : ty_ChainerConcat(F.vstack),
        }


ext_callable_ty = {
        L.Linear             : ty_ChainerLinear(),
        L.Convolution2D      : ty_ChainerConvolution2D(),
        L.BatchNormalization : ty_ChainerBatchNormalization,
        L.EmbedID            : ty_ChainerEmbedID,
        L.NStepBiLSTM        : ty_ChainerNStepBiLSTM,
        }
