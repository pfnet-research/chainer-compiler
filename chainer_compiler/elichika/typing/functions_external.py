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
        return value_of_type(ty_kwargs[key]), is_dummy_value(ty_kwargs[key])
    return default, False

def make_infer(func, fallback_shapes, fallback_dtypes):
    def infer(ty_args, ty_kwargs):
        ty_args_tensor = [t for t in ty_args if isinstance(t, TyTensor)]

        shapes = [s if t.shape is None else t.shape
                for t, s in zip(ty_args_tensor, fallback_shapes)]
        dtypes = [dt if t.dtype.t is None else t.dtype.t
                for t, dt in zip(ty_args_tensor, fallback_dtypes)]
        is_dummy_shape = any([t.shape is None for t in ty_args_tensor])
        is_dummy_dtype = any([t.dtype.t is None for t in ty_args_tensor])

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
                ty_result.dtype.t = None
        return ty_result

    return infer


# 'evaluate' function return type by using the function
def evaluate_function_types(func, narg_tensor=None, fallback_shapes=None, fallback_dtypes=None):
    assert narg_tensor is not None or \
            fallback_shapes is not None and fallback_dtypes is not None
    if fallback_shapes is None:
        fallback_shapes = ((1, 1),) * narg_tensor
    if fallback_dtypes is None:
        fallback_dtypes = (np.float32,) * narg_tensor

    return make_infer(func, fallback_shapes, fallback_dtypes)


def ty_NumpyOnes(ty_args, ty_kwargs):
    is_dummy_shape = is_dummy_value(ty_args[0])
    shape = value_of_type(ty_args[0])
    dtype, is_dummy_dtype = get_kwarg(ty_kwargs, 'dtype', None)
    if dtype is None:
        dtype = np.dtype('float64')
    ty_ret = TyNdarray(dtype=TyDType(dtype), shape=shape)
    if is_dummy_shape:
        ty_ret.shape = None
    if is_dummy_dtype:
        ty_ret.dtype.t = None
    return ty_ret


def ty_ChainerPooling2d(func):
    def infer(ty_args, ty_kwargs):
        ksize = ty_args[1].value  # cannot be None
        # TODO(momohatt): use is_dummy_stride
        stride, is_dummy_stride = get_kwarg(ty_kwargs, 'stride', default=ksize)
        minimum_size = max(ksize, stride)
        fallback_shapes = ((1, 1, minimum_size, minimum_size),)
        fallback_dtypes = (np.float32,)

        return make_infer(func, fallback_shapes, fallback_dtypes) \
                (ty_args, ty_kwargs)

    return infer


def ty_ChainerSoftmaxCrossEntropy(ty_args, ty_kwargs):
    shape_x, shape_t = ty_args[0].shape, ty_args[1].shape
    fallback_dtypes = (np.float32, np.int64)

    # x.shape[0] == t.shape[0]
    if shape_x is None and shape_t is not None:
        fallback_shapes = ((shape_t[0], 1), shape_t)
    elif shape_x is not None and shape_t is None:
        fallback_shapes = (shape_x, (shape_x[0],))
    else:
        fallback_shapes = ((1, 1), (1,))

    return make_infer(
            F.softmax_cross_entropy, fallback_shapes, fallback_dtypes) \
                    (ty_args, ty_kwargs)


# math functions that doesn't change shapes or dtypes
def ty_ChainerIdentical(is_float_only=True):
    def infer(ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TyTensor)
        if is_float_only:
            assert ty_args[0].dtype.is_float()
        return ty_args[0]

    return infer


def ty_ChainerConcat(func):
    def infer(ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TySequence)
        if not ty_args[0].is_fixed_len:
            dtype = ty_args[0].get_ty().dtype
            return TyChainerVariable(dtype=dtype)

        dtypes = [tytensor.dtype for tytensor in ty_args[0].get_tys()]
        assert all_same(dtypes)

        if is_dummy_value(ty_args[0]):
            return TyChainerVariable(dtype=dtypes[0])

        dummy_xs = value_of_type(ty_args[0])

        if func is F.vstack or func is F.hstack:
            return type_of_value(func(dummy_xs))

        # TODO(momohatt): use is_dummy_axis
        axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', default=1)
        return type_of_value(func(dummy_xs, axis=axis))

    return infer


def ty_ChainerExpandDims(ty_args, ty_kwargs):
    # TODO(momohatt): axis can come as ty_kwargs
    axis = ty_args[1].value if len(ty_args) == 2 else ty_kwargs['axis'].value
    fallback_shapes = ((1,) * axis,)
    fallback_dtypes = (np.float32,)

    return make_infer(F.expand_dims, fallback_shapes, fallback_dtypes) \
            (ty_args, ty_kwargs)


def ty_ChainerBroadcastTo(ty_args, ty_kwargs):
    if ty_args[0].shape is None or is_dummy_value(ty_args[1]):
        return TyChainerVariable(dtype=ty_args[0].dtype)

    in_shape = ty_args[0].shape
    out_shape = value_of_type(ty_args[1])

    # check_type_forward
    ndim = len(out_shape)
    assert len(in_shape) <= ndim

    for i in range(-1, - len(in_shape) - 1, -1):
        assert in_shape[i] == out_shape[i] or in_shape[i] == 1

    return TyChainerVariable(dtype=ty_args[0].dtype, shape=out_shape)


def ty_ChainerSum(ty_args, ty_kwargs):
    # TODO(momohatt): use is_dummy_axis
    axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', default=None)
    fallback_shapes = ((1,) * (axis + 1),)
    fallback_dtypes = (np.float32,)

    return make_infer(F.sum, fallback_shapes, fallback_dtypes) \
            (ty_args, ty_kwargs)


def calculate_reshape(orig_shape, input_shape):
    # orig_shape can be None
    # input_shape should be accurate
    if orig_shape is None:
        if any([i == -1 for i in input_shape]):
            return None
        return input_shape
    fill = abs(size_of_shape(orig_shape) // size_of_shape(input_shape))
    return tuple([i if i != -1 else fill for i in input_shape])


def ty_ChainerReshape(ty_args, ty_kwargs):
    dtype = ty_args[0].dtype
    shape = ty_args[0].shape

    if is_dummy_value(ty_args[1]):
        return TyChainerVariable(dtype=dtype, shape=None)
    ret_shape = calculate_reshape(shape, value_of_type(ty_args[1]))
    return TyChainerVariable(dtype=dtype, shape=ret_shape)


def ty_ChainerSqueeze(ty_args, ty_kwargs):
    # TODO: don't use F.squeeze
    axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', None)

    if ty_args[0].shape is None or is_dummy_value(ty_args[0]) or is_dummy_axis:
        return TyChainerVariable(dtype=ty_args[0].dtype)

    ret = F.squeeze(value_of_type(ty_args[0]), axis=axis)
    return type_of_value(ret)


def ty_ChainerSwapAxes(ty_args, ty_kwargs):
    def swap(t, i, j):
        l = list(t)
        l[i], l[j] = l[j], l[i]
        return tuple(l)


    if ty_args[0].shape is None or \
            is_dummy_value(ty_args[1]) or is_dummy_value(ty_args[2]):
        return TyChainerVariable(dtype=ty_args[0].dtype)

    shape = ty_args[0].shape
    axis1 = value_of_type(ty_args[1])
    axis2 = value_of_type(ty_args[2])

    assert axis1 < len(shape) and axis2 < len(shape)

    return TyChainerVariable(dtype=ty_args[0].dtype,
            shape=swap(shape, axis1, axis2))


def ty_ChainerSeparate(ty_args, ty_kwargs):
    axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', 0)
    assert isinstance(ty_args[0], TyTensor)
    x_shape = ty_args[0].shape
    x_dtype = ty_args[0].dtype

    if x_shape is None or is_dummy_axis:
        return TyTuple(TyChainerVariable(dtype=x_dtype))

    assert axis < len(x_shape)

    n = x_shape[axis]
    out_shape = x_shape[:axis] + x_shape[axis + 1:]
    return TyTuple([TyChainerVariable(dtype=x_dtype, shape=out_shape)] * n)


def ty_ChainerSplitAxis(ty_args, ty_kwargs):
    assert isinstance(ty_args[0], TyTensor)

    if isinstance(ty_args[1], TyNum):
        n = ty_args[1].value
        if n is None:
            # TODO
            return TyVar()
        return TyTuple([TyChainerVariable(dtype=ty_args[0].dtype)] * n)

    assert isinstance(ty_args[1], TyTensor)
    # 1-D array

    if ty_args[1].shape is None:
        # variable length tuple
        return TyTuple(TyChainerVariable(dtype=ty_args[0].dtype))
    n = ty_args[1].shape[0]
    return TyTuple([TyChainerVariable(dtype=ty_args[0].dtype)] * n)


def ty_ChainerPadSequence(ty_args, ty_kwargs):
    # shape[1:] should be uniform & ndim > 0
    ty = ty_args[0]
    assert isinstance(ty, TySequence)
    is_dummy_shape = False
    if ty.is_fixed_len:
        is_shape_None = [t.shape is None for t in ty.get_tys()]
        if any(is_shape_None):
            is_dummy_shape = True
            if all(is_shape_None):
                dummy_arg = [np.zeros((1,)) for _ in ty.get_tys()]
            else:
                t = utils.find(ty.get_tys(), lambda t: t.shape is not None)
                fallback_shape = t.shape
                dummy_arg = [
                        np.zeros(fallback_shape) if t.shape is None
                        else np.zeros(t.shape) for t in ty.get_tys()]
        else:
            dummy_arg = value_of_type(ty_args[0])
    else:
        is_dummy_shape = True
        dummy_arg = [np.zeros((1,), dtype=ty.get_ty().dtype.t)]

    dummy_kwargs = {k : value_of_type(t) for (k, t) in ty_kwargs.items()}
    ty_ret = type_of_value(F.pad_sequence(dummy_arg, **dummy_kwargs))
    if is_dummy_shape:
        ty_ret.shape = None
    return ty_ret



class ty_ChainerLinear():
    def __call__(self, linear, ty_args, ty_kwargs):
        shape = ty_args[0].shape
        dtype = ty_args[0].dtype
        n_batch_axes, is_dummy_n_batch_axes = \
                get_kwarg(ty_kwargs, 'n_batch_axes', default=1)

        if dtype.t is not None and linear.b is not None:
            assert dtype.t == linear.b.dtype
        if shape is None or is_dummy_n_batch_axes:
            return TyChainerVariable(dtype=dtype, shape=None)

        out_shape = self.calculate_return_shape(linear, shape, n_batch_axes)
        return TyChainerVariable(dtype=dtype, shape=out_shape)

    def calculate_return_shape(self, linear, x_shape, n_batch_axes):
        assert n_batch_axes >= 1

        if n_batch_axes > 1:
            batch_shape = x_shape[:n_batch_axes]
            batch_size = size_of_shape(batch_shape)
            x_shape = calculate_reshape(x_shape, (batch_size, -1))
        elif len(x_shape) > 2:
            x_shape = calculate_reshape(x_shape, (x_shape[0], -1))

        if linear.in_size is not None:
            assert x_shape[1] == linear.in_size

        y_shape = (x_shape[0], linear.out_size)
        if n_batch_axes > 1:
            y_shape = calculate_reshape(y_shape, (batch_shape + (-1,)))
        return y_shape


def ty_ChainerConvolution2D(conv, ty_args, ty_kwargs):
    shape = ty_args[0].shape
    dtype = ty_args[0].dtype
    if dtype.t is not None:
        assert dtype.is_float()
    if shape is None:
        return TyChainerVariable(dtype=dtype, shape=None)

    assert len(shape) == 4
    if conv.in_channels is not None:
        assert shape[1] == conv.in_channels

    ksize = make_pair(conv.ksize)
    stride = make_pair(conv.stride)
    pad = make_pair(conv.pad)
    dilate = make_pair(conv.dilate)

    shape_2 = get_conv_outsize(
            shape[2], ksize[0], stride[0], pad[0], d=dilate[0])
    shape_3 = get_conv_outsize(
            shape[3], ksize[1], stride[1], pad[1], d=dilate[1])
    return TyChainerVariable(dtype=dtype,
            shape=(shape[0], conv.out_channels, shape_2, shape_3))


def ty_ChainerBatchNormalization(obj, ty_args, ty_kwargs):
    assert False


def ty_ChainerEmbedID(embed, ty_args, ty_kwargs):
    assert isinstance(ty_args[0], TyTensor)
    # TODO(momohatt): align naming rules of variables
    x_type = ty_args[0]
    w_type = embed.W

    if x_type.shape is None:
        return TyChainerVariable(dtype=TyDType(w_type.dtype))

    assert x_type.dtype.is_int()
    assert len(x_type.shape) >= 1

    assert all([t < w_type.shape[0] for t in x_type.shape])
    out_type = x_type.shape + (w_type.shape[1],)
    return TyChainerVariable(dtype=w_type.dtype, shape=out_shape)


def ty_ChainerNStepBiLSTM(nblstm, ty_args, ty_kwargs):
    ty_hx = ty_args[0]
    ty_cx = ty_args[1]
    ty_xs = ty_args[2]
    assert isinstance(ty_xs, TySequence)

    if not ty_xs.is_fixed_len:
        # TODO
        return TyTuple([ty_hx, ty_cx, ty_xs])

    xs_dtypes = [t.dtype for t in ty_xs.get_tys()]
    xs_shapes = [t.shape for t in ty_xs.get_tys()]
    assert all_same(xs_dtypes)

    if isinstance(ty_hx, TyTensor):
        hx_shape = ty_hx.shape
        hx_dtype = ty_hx.dtype
    else:
        hx_shape = (nblstm.n_layers * 2, len(ty_xs.get_tys()), nblstm.out_size)
        hx_dtype = ty_xs.get_tys()[0].dtype

    if isinstance(ty_cx, TyTensor):
        cx_shape = ty_cx.shape
        cx_dtype = ty_cx.dtype
    else:
        cx_shape = (nblstm.n_layers * 2, len(ty_xs.get_tys()), nblstm.out_size)
        cx_dtype = hx_dtype

    if hx_shape is None or cx_shape is None:
        # TODO: 適当
        return TyTuple([ty_hx, ty_cx, ty_xs])

    assert hx_shape[0] // 2 == nblstm.n_layers
    assert hx_shape == cx_shape
    N = hx_shape[2]

    ty_hy = TyChainerVariable(dtype=hx_dtype, shape=hx_shape)
    ty_cy = TyChainerVariable(dtype=cx_dtype, shape=cx_shape)

    if any([t.shape is None for t in ty_xs.get_tys()]):
        return TyTuple([ty_hy, ty_cy, TyList(TyChainerVariable(dtype=xs_dtypes[0]))])

    # TODO(momohatt): nblstm doesn't have attribute in_size
    # assert all([i == nblstm.in_size for _, i in xs_shapes])
    ys_shapes = [(l, 2 * N) for l, _ in xs_shapes]
    ty_ys = TyList([TyChainerVariable(dtype=xs_dtypes[0], shape=s) for s in ys_shapes])

    return TyTuple([ty_hy, ty_cy, ty_ys])


ext_func_ty = {
        np.array : evaluate_function_types(
            np.array, 0),
        np.cumsum :
            ty_ChainerIdentical(is_float_only=False),
        np.full : evaluate_function_types(
            np.full, 0),
        np.ones :
            ty_NumpyOnes,
        np.zeros :
            ty_NumpyOnes,
        chainer.Variable : evaluate_function_types(
            chainer.Variable, 1),
        cuda.to_cpu :
            ty_ChainerIdentical(is_float_only=False),
        F.average_pooling_2d :
            ty_ChainerPooling2d(F.average_pooling_2d),
        F.broadcast_to :
            ty_ChainerBroadcastTo,
        F.concat :
            ty_ChainerConcat(F.concat),
        F.dropout :
            ty_ChainerIdentical(),
        F.expand_dims :
            ty_ChainerExpandDims,
        F.hstack :
            ty_ChainerConcat(F.hstack),
        F.local_response_normalization : evaluate_function_types(
            F.local_response_normalization, 1),
        F.max_pooling_2d :
            ty_ChainerPooling2d(F.max_pooling_2d),
        F.pad_sequence :
            ty_ChainerPadSequence,
        F.relu :
            ty_ChainerIdentical(),
        F.reshape :
            ty_ChainerReshape,
        F.separate :
            ty_ChainerSeparate,
        F.sigmoid :
            ty_ChainerIdentical(),
        F.split_axis :
            ty_ChainerSplitAxis,
        F.squeeze :
            ty_ChainerSqueeze,
        F.softmax :
            ty_ChainerIdentical(),
        F.softmax_cross_entropy :
            ty_ChainerSoftmaxCrossEntropy,
        F.stack :
            ty_ChainerConcat(F.stack),
        F.sum :
            ty_ChainerSum,
        F.swapaxes :
            ty_ChainerSwapAxes,
        F.tanh :
            ty_ChainerIdentical(),
        F.vstack :
            ty_ChainerConcat(F.vstack),
        }


ext_callable_ty = {
        L.Linear : ty_ChainerLinear(),
        L.Convolution2D : ty_ChainerConvolution2D,
        L.BatchNormalization : ty_ChainerBatchNormalization,
        L.EmbedID : ty_ChainerEmbedID,
        L.NStepBiLSTM : ty_ChainerNStepBiLSTM,
        }


