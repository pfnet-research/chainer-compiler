import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer.utils.conv import get_conv_outsize

from chainer_compiler.elichika.typing.types import *


def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    return x

def get_kwarg(ty_kwargs, key, default=None):
    if key in ty_kwargs.keys():
        return value_of_type(ty_kwargs[key])
    return default

def make_infer(func, fallback_shapes, fallback_dtypes):
    def infer(ty_args, ty_kwargs):
        ty_args_tensor = [t for t in ty_args if isinstance(t, TyTensor)]

        shapes = [s if t.shape is None else t.shape
                for t, s in zip(ty_args_tensor, fallback_shapes)]
        dtypes = [s if t.dtype.t is None else t.dtype.t
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


def ty_ChainerPooling2d(func):
    def infer(ty_args, ty_kwargs):
        ksize = ty_args[1].value  # cannot be None
        stride = get_kwarg(ty_kwargs, 'stride', default=ksize)
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
        if isinstance(ty_args[0], TyTensor):
            if is_float_only:
                assert ty_args[0].dtype.is_float()
            return ty_args[0]
        assert False

    return infer


def ty_ChainerConcat(ty_args, ty_kwargs):
    # TODO(momohatt): shape
    assert isinstance(ty_args[0], TySequence)
    if ty_args[0].is_fixed_len:
        dtypes = [tytensor.dtype for tytensor in ty_args[0].get_tys()]
        assert all_same(dtypes)
        return TyChainerVariable(dtype=dtypes[0])

    dtype = ty_args[0].get_ty().dtype
    return TyChainerVariable(dtype=dtype)


def ty_ChainerExpandDims(ty_args, ty_kwargs):
    # TODO(momohatt): axis can come as ty_kwargs
    axis = ty_args[1].value if len(ty_args) == 2 else ty_kwargs['axis'].value
    fallback_shapes = ((1,) * axis,)
    fallback_dtypes = (np.float32,)

    return make_infer(F.expand_dims, fallback_shapes, fallback_dtypes) \
            (ty_args, ty_kwargs)


def ty_ChainerBroadcastTo(ty_args, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSum(ty_args, ty_kwargs):
    axis = get_kwarg(ty_kwargs, 'axis', default=None)
    fallback_shapes = ((1,) * (axis + 1),)
    fallback_dtypes = (np.float32,)

    return make_infer(F.sum, fallback_shapes, fallback_dtypes) \
            (ty_args, ty_kwargs)


def ty_ChainerReshape(ty_args, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype,
            shape=value_of_type(ty_args[1]))


def ty_ChainerSqueeze(ty_args, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSwapAxes(ty_args, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSeparate(ty_args, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSplitAxis(ty_args, ty_kwargs):
    assert isinstance(ty_args[0], TyTensor)

    if isinstance(ty_args[1], TyNum):
        n = ty_args[1].value
        return TyTuple([TyChainerVariable(dtype=ty_args[0].dtype)] * n)
    elif isinstance(ty_args[1], TyTensor):
        # 1-D array
        if ty_args[1].shape is None:
            # variable length tuple
            return TyTuple(TyChainerVariable(dtype=ty_args[0].dtype))
        n = ty_args[1].shape[0]
        return TyTuple([TyChainerVariable(dtype=ty_args[0].dtype)] * n)

    assert False


def ty_ChainerPadSequence(ty_args, ty_kwargs):
    # shape[1:] should be uniform & ndim > 0
    ty = ty_args[0].deref()
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
        if ty.get_ty().shape is None:
            is_dummy_shape = True
            dummy_arg = [np.zeros((1,), dtype=ty.get_ty().dtype.t)]

    dummy_kwargs = {k : value_of_type(t) for (k, t) in ty_kwargs.items()}
    ty_ret = type_of_value(F.pad_sequence(dummy_arg, **dummy_kwargs))
    if is_dummy_shape:
        ty_ret.shape = None
    return ty_ret


def ty_ChainerLinear(obj, ty_args, ty_kwargs):
    shape = ty_args[0].shape
    dtype = ty_args[0].dtype

    if dtype.t is not None:
        assert dtype.t == obj.b.dtype
    if shape is None:
        return TyChainerVariable(dtype=dtype, shape=None)

    if len(shape) > 2:
        # TODO: case of reshape
        pass
    assert len(shape) == 2
    if obj.in_size is not None:
        assert shape[1] == obj.in_size

    return TyChainerVariable(dtype=dtype, shape=(shape[0], obj.out_size))


def ty_ChainerConvolution2D(obj, ty_args, ty_kwargs):
    shape = ty_args[0].shape
    dtype = ty_args[0].dtype
    if dtype.t is not None:
        assert dtype.is_float()
    if shape is None:
        return TyChainerVariable(dtype=dtype, shape=None)

    assert len(shape) == 4
    if obj.in_channels is not None:
        assert shape[1] == obj.in_channels

    ksize = make_pair(obj.ksize)
    stride = make_pair(obj.stride)
    pad = make_pair(obj.pad)
    dilate = make_pair(obj.dilate)

    shape_2 = get_conv_outsize(
            shape[2], ksize[0], stride[0], pad[0], d=dilate[0])
    shape_3 = get_conv_outsize(
            shape[3], ksize[1], stride[1], pad[1], d=dilate[1])
    return TyChainerVariable(dtype=dtype,
            shape=(shape[0], obj.out_channels, shape_2, shape_3))


def ty_ChainerBatchNormalization(obj, ty_args, ty_kwargs):
    assert False

def ty_ChainerEmbedID(obj, ty_args, ty_kwargs):
    assert False

def ty_ChainerNStepBiLSTM(obj, ty_args, ty_kwargs):
    assert False


ext_func_ty = {
        np.array : evaluate_function_types(
            np.array, 0),
        np.cumsum :
            ty_ChainerIdentical(is_float_only=False),
        np.full : evaluate_function_types(
            np.full, 0),
        np.ones : evaluate_function_types(
            np.ones, 0),
        np.zeros : evaluate_function_types(
            np.zeros, 0),
        chainer.Variable : evaluate_function_types(
            chainer.Variable, 1),
        cuda.to_cpu :
            ty_ChainerIdentical(is_float_only=False),
        F.average_pooling_2d :
            ty_ChainerPooling2d(F.average_pooling_2d),
        F.broadcast_to :
            ty_ChainerBroadcastTo,
        F.concat :
            ty_ChainerConcat,
        F.dropout :
            ty_ChainerIdentical(),
        F.expand_dims :
            ty_ChainerExpandDims,
        F.hstack :
            ty_ChainerConcat,
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
            ty_ChainerConcat,
        F.sum :
            ty_ChainerSum,
        F.swapaxes :
            ty_ChainerSwapAxes,
        F.tanh :
            ty_ChainerIdentical(),
        F.vstack :
            ty_ChainerConcat,
        }


ext_callable_ty = {
        L.Linear : ty_ChainerLinear,
        L.Convolution2D : ty_ChainerConvolution2D,
        L.BatchNormalization : ty_ChainerBatchNormalization,
        L.EmbedID : ty_ChainerEmbedID,
        L.NStepBiLSTM : ty_ChainerNStepBiLSTM,
        }


