import chainer
from   chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import math

import six
from   typing import List

from   chainer.utils.conv import get_conv_outsize
from   chainer.utils import size_of_shape
from   chainer.utils import type_check

from   chainer_compiler.elichika.typing.types import *

def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    return x


def get_kwarg(ty_kwargs, key, default=None):
    if key in ty_kwargs.keys():
        # TODO(momohatt): when unable to get the correct value, do something
        return value_of_type(ty_kwargs[key]), lacks_value(ty_kwargs[key])
    return default, False


def make_multiple_tc_variable(ty_args, names):
    assert len(ty_args) == len(names)
    return [type_check.Variable(t, n) for t, n in zip(ty_args, names)]


def make_sequence_tc_Variable(ty_arg, name):
    ret = []
    for i in range(ty_arg.size()):
        ret.append(type_check.Variable(ty_arg[i], '{}[{}]'.format(name, i)))
    return ret


def calculate_reshape(orig_shape, input_shape):
    if is_incomplete_shape(orig_shape):
        if any([i == -1 for i in input_shape]):
            return None
        return input_shape
    fill = abs(size_of_shape(orig_shape) // size_of_shape(input_shape))
    ret_shape = tuple([i if i != -1 else fill for i in input_shape])
    assert size_of_shape(orig_shape) == size_of_shape(ret_shape)
    return wrap_shape(ret_shape)


# TODO(momohatt): use chainer.utils.type_check.expect

def infer_return_type(inference_logic, args, is_fake_shape, is_fake_dtype):
    ty_result = inference_logic(*args)

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


def make_infer(func, fallback_shapes, fallback_dtypes):  # 丸投げ
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


class ty_NumpyArray():
    def __call__(self, ty_args, ty_kwargs):
        infer = make_infer(np.array, (), ())
        return infer(ty_args, ty_kwargs)


class ty_NumpyOnes():
    def __call__(self, ty_args, ty_kwargs):
        shape_type = ty_args[0]
        is_dummy_shape = lacks_value(shape_type)
        shape = value_of_type(shape_type)
        dtype, is_dummy_dtype = get_kwarg(ty_kwargs, 'dtype', np.dtype('float64'))

        ty_ret = TyNdarray(dtype, shape=shape)
        if is_dummy_shape: ty_ret.shape = None
        if is_dummy_dtype: ty_ret.dtype = None
        return ty_ret


class ty_NumpyFull():
    def __call__(self, ty_args, ty_kwargs):
        shape_type = ty_args[0]
        value_type = ty_args[1]
        dtype, is_dummy_dtype = get_kwarg(ty_kwargs, 'dtype', tyobj2dtype(value_type))
        assert not is_dummy_dtype

        assert isinstance(shape_type, TyNum) or isinstance(shape_type, TyTuple)
        is_dummy_shape = lacks_value(shape_type)
        shape = value_of_type(shape_type)
        if isinstance(shape, int):
            shape = (shape, )

        y_type = TyNdarray(dtype, shape=wrap_shape(shape))
        if is_dummy_shape: y_type.shape = None
        return y_type


class ty_ChainerVariable():
    def __call__(self, ty_args, ty_kwargs):
        infer = make_infer(chainer.Variable, (1,), (np.float32,))
        return infer(ty_args, ty_kwargs)


class ty_ChainerPooling2d():
    # max_pooling_2d / average_pooling_2d
    def __call__(self, ty_args, ty_kwargs):
        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x', 'ksize')))

        x_type = ty_args[0]

        # TODO(momohatt): handle cases where ksize is unknown
        ksize = ty_args[1].value
        # TODO(momohatt): use is_dummy_stride
        stride, is_dummy_stride = get_kwarg(ty_kwargs, 'stride', default=ksize)
        pad, is_dummy_pad = get_kwarg(ty_kwargs, 'pad', default=0)

        return self.infer_return(x_type, ksize, stride, pad)

    def check_type_forward(self, in_types: List['type_check.Variable']):
        x_type = in_types[0]

        type_check.expect(
                x_type.dtype.kind == 'f',
                )

        # assert isinstance(ksize_type, TyNum)

    def infer_return(self, x_type, ksize, stride, pad):
        pad = make_pair(pad)
        ksize = make_pair(ksize)
        stride = make_pair(stride)

        shape_0 = x_type.shape[0]
        shape_1 = x_type.shape[1]
        shape_2 = math.ceil((x_type.shape[2] + pad[0] * 2 - ksize[0]) / stride[0]) + 1
        shape_3 = math.ceil((x_type.shape[3] + pad[1] * 2 - ksize[1]) / stride[1]) + 1

        return TyChainerVariable(x_type.dtype,
                shape=(shape_0, shape_1, shape_2, shape_3))


class ty_ChainerSoftmaxCrossEntropy():
    def __call__(self, ty_args, ty_kwargs):
        x_type, t_type = ty_args
        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x', 't')))
        return self.infer_return(x_type, t_type)

    def check_type_forward(self, in_types):
        x_type, t_type = in_types

        type_check.expect(
                x_type.dtype.kind == 'f',
                t_type.dtype.kind == 'i',
                t_type.ndim == x_type.ndim - 1,
                x_type.shape[0] == t_type.shape[0],
                x_type.shape[2:] == t_type.shape[1:]
                )

    def infer_return(self, x_type, t_type):
        return TyChainerVariable(x_type.dtype, shape=())


class ty_ChainerIdentical():
    # functions that doesn't change shapes or dtypes

    def __init__(self, is_float_only=True):
        self.is_float_only = is_float_only

    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        if self.is_float_only:
            assert x_type.dtype.kind == 'f'
        return copy_ty(x_type)


class ty_ChainerConcat():
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, ty_args, ty_kwargs):
        xs_type = ty_args[0]

        assert isinstance(xs_type, TySequence)
        self.axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', default=1)

        if lacks_value(xs_type) or is_dummy_axis:
            x_type = xs_type.get()
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.make_variable(self.axis, 'axis'))

        type_check.expect(
            -in_types[0].ndim <= self.axis,
            self.axis < in_types[0].ndim
        )
        ndim = type_check.eval(in_types[0].ndim)
        axis = self.axis % ndim
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in six.moves.range(0, ndim):
                if d == axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        ret_shape = list(xs_type[0].shape)
        ret_shape[self.axis] = sum([x_type.shape[self.axis] for x_type in xs_type])
        return TyChainerVariable(dtype=xs_type[0].dtype,
                shape=wrap_shape(ret_shape))


class ty_ChainerStack():
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, ty_args, ty_kwargs):
        xs_type = ty_args[0]

        assert isinstance(xs_type, TySequence)
        self.axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', default=1)

        if lacks_value(xs_type) or is_dummy_axis:
            x_type = xs_type.get()
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim + 1)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        self.axis = xs_type.get().ndim + 1 - abs(self.axis)
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(
            -in_types[0].ndim - 1 <= self.axis,
            self.axis <= in_types[0].ndim
        )

        # XXX: modified
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].shape == in_types[i].shape,
            )

        # XXX: the following doesn't work
        # dtype = in_types[0].dtype
        # shape = in_types[0].shape
        # for x_type in in_types[1:]:
        #     type_check.expect(
        #         x_type.dtype == dtype,
        #         x_type.shape == shape,
        #     )

    def infer_return(self, xs_type):
        ret_shape = list(xs_type[0].shape)
        ret_shape.insert(self.axis, len(xs_type))
        return TyChainerVariable(xs_type.get().dtype,
                shape=wrap_shape(ret_shape))


class ty_ChainerHstack():
    def __call__(self, ty_args, ty_kwargs):
        xs_type = ty_args[0]

        assert isinstance(xs_type, TySequence)
        if lacks_value(xs_type):
            x_type = xs_type.get()
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check._argname((in_types[0],), ('x0',))

        ndim = type_check.eval(in_types[0].ndim)
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check._argname((in_types[i],), ('x{}'.format(i),))
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                continue
            for d in six.moves.range(0, ndim):
                if d == 1:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        if xs_type[0].ndim <= 1:
            return ty_ChainerConcat(axis=0).infer_return(xs_type)
        return ty_ChainerConcat(axis=1).infer_return(xs_type)


class ty_ChainerVstack():
    def __call__(self, ty_args, ty_kwargs):
        xs_type = ty_args[0]

        assert isinstance(xs_type, TySequence)
        if lacks_value(xs_type):
            x_type = xs_type.get()
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = type_check.eval(in_types[0].ndim)
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in six.moves.range(1, ndim):
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        if xs_type[0].ndim <= 1:
            return ty_ChainerStack(axis=0).infer_return(xs_type)
        return ty_ChainerConcat(axis=0).infer_return(xs_type)


class ty_ChainerExpandDims():
    def __call__(self, ty_args, ty_kwargs):
        axis = ty_args[1].value
        fallback_shapes = (wrap_shape((1,) * axis,))
        fallback_dtypes = (np.float32,)

        infer = make_infer(F.expand_dims, fallback_shapes, fallback_dtypes)
        return infer(ty_args, ty_kwargs)


class ty_ChainerBroadcastTo():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        shape_type = ty_args[1]

        assert shape_type.is_fixed_len

        if x_type.shape is None or lacks_value(shape_type):
            return TyChainerVariable(x_type.dtype, ndim=len(shape_type.get_tys()))

        in_shape = x_type.shape
        out_shape = wrap_shape(value_of_type(shape_type))

        # check_type_forward
        ndim = len(out_shape)
        assert len(in_shape) <= ndim

        for i in range(-1, - len(in_shape) - 1, -1):
            assert in_shape[i] == out_shape[i] or in_shape[i] == 1

        return TyChainerVariable(x_type.dtype, shape=out_shape)


class ty_ChainerSum():
    def __call__(self, ty_args, ty_kwargs):
        # TODO(momohatt): use is_dummy_axis
        axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', default=None)
        fallback_shapes = (wrap_shape((1,) * (axis + 1)),)
        fallback_dtypes = (np.float32,)

        infer = make_infer(F.sum, fallback_shapes, fallback_dtypes)
        return infer(ty_args, ty_kwargs)


class ty_ChainerReshape():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_type = ty_args

        assert shape_type.is_fixed_len

        if lacks_value(shape_type):
            # TODO: ndim
            return TyChainerVariable(x_type.dtype,
                    ndim=len(shape_type.get_tys()))

        return self.infer_return(x_type, value_of_type(shape_type))

    def infer_return(self, x_type, shape):
        ret_shape = calculate_reshape(x_type.shape, shape)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerRepeat():
    def __init__(self):
        self.axis = None

    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        repeats_type = ty_args[1]
        self.axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', 0)

        if lacks_value(repeats_type) or is_dummy_axis:
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        repeats = value_of_type(repeats_type)
        self.check_type_forward(x_type, repeats)
        return self.infer_return(x_type, repeats)

    def check_type_forward(self, x_type, repeats):
        # XXX: This is not taken from chainer nor numpy
        if isinstance(repeats, int):
            assert self.axis < x_type.ndim
            return
        assert x_type.shape[self.axis] == len(repeats), "repeat"


    def infer_return(self, x_type, repeats):
        if isinstance(repeats, int):
            if x_type.ndim < 1:
                ret_shape = (repeats,)
            else:
                ret_shape = list(x_type.shape)
                ret_shape[self.axis] = x_type.shape[self.axis] * repeats
        else:
            ret_shape = list(x_type.shape)
            ret_shape[self.axis] = sum(repeats)
        return TyChainerVariable(x_type.dtype, shape=wrap_shape(ret_shape))


class ty_ChainerSqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]

        # TODO: don't use F.squeeze
        axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', None)

        if x_type.shape is None or lacks_value(x_type) or is_dummy_axis:
            return TyChainerVariable(x_type.dtype, shape=())  # TODO

        ret = F.squeeze(value_of_type(x_type), axis=axis)
        return type_of_value(ret)


class ty_ChainerSwapAxes():
    def __call__(self, ty_args, ty_kwargs):
        def swap(t, i, j):
            l = list(t)
            l[i], l[j] = l[j], l[i]
            return tuple(l)

        x_type = ty_args[0]
        axis1_type = ty_args[1]
        axis2_type = ty_args[2]

        if x_type.shape is None or \
                lacks_value(axis1_type) or lacks_value(axis2_type):
            return TyChainerVariable(x_type.dtype)

        shape = x_type.shape
        axis1 = value_of_type(axis1_type)
        axis2 = value_of_type(axis2_type)

        assert axis1 < len(shape) and axis2 < len(shape)

        return TyChainerVariable(x_type.dtype,
                shape=swap(shape, axis1, axis2))


class ty_ChainerSeparate():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        axis, is_dummy_axis = get_kwarg(ty_kwargs, 'axis', 0)
        assert isinstance(x_type, TyTensor)
        x_shape = x_type.shape
        x_dtype = x_type.dtype

        if x_shape is None or is_dummy_axis:
            return TyTuple(TyChainerVariable(x_dtype, shape=())) # TODO

        assert axis < len(x_shape)

        n = x_shape[axis]
        out_shape = x_shape[:axis] + x_shape[axis + 1:]
        return TyTuple([TyChainerVariable(x_dtype, shape=out_shape)] * n)


class ty_ChainerSplitAxis():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]

        assert isinstance(x_type, TyTensor)

        if isinstance(ty_args[1], TyNum):
            n = ty_args[1].value
            if n is None:
                # TODO
                return TyVar()
            return TyTuple([TyChainerVariable(x_type.dtype, shape=())] * n)  # TODO

        assert isinstance(ty_args[1], TyTensor)
        # 1-D array

        if ty_args[1].shape is None:
            # variable length tuple
            return TyTuple(TyChainerVariable(x_type.dtype, shape=()))  # TODO

        assert len(ty_args[1].shape) == 1
        n = int(ty_args[1].shape[0])
        return TyTuple([TyChainerVariable(x_type.dtype, shape=())] * n) # TODO


class ty_ChainerPad():
    def __call__(self, ty_args, ty_kwargs):
        x_type, pad_width_type, mode_type = ty_args

        assert isinstance(mode_type, TyString), \
                "chainer.functions.pad: mode_type should be string"
        self.check_type_forward(make_multiple_tc_variable(ty_args[:1], ('x',)))

        if lacks_value(pad_width_type):
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        assert pad_width_type.size() > 0, \
                "chainer.functions.pad: pad_width is not specified"

        pad_width = value_of_type(pad_width_type)
        if isinstance(pad_width, int):
            pad_width = make_pair(pad_width)
        if isinstance(pad_width[0], int):
            pad_width = pad_width * x_type.ndim
        for pad in pad_width:
            assert len(pad) == 2, "chainer.functions.pad: pad_width is invalid"
        return self.infer_return(x_type, pad_width)

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

    def infer_return(self, x_type, pad_width):
        ret_shape = list(x_type.shape)
        for i in range(x_type.ndim):
            ret_shape[i] += pad_width[i][0] + pad_width[i][1]
        return TyChainerVariable(x_type.dtype, shape=wrap_shape(ret_shape))


class ty_ChainerPadSequence():
    def __call__(self, ty_args, ty_kwargs):
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


class ty_ChainerLocalResponseNormalization():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]

        self.check_type(x_type)
        return self.infer_return(x_type)

    def check_type(self, x_type):
        x = type_check.Variable(x_type, 'x')

        type_check.expect(
            x.dtype.kind == 'f',
            x.ndim >= 2,
        )
        assert len(x_type.shape) >= 2

    def infer_return(self, x_type):
        return TyChainerVariable(
                dtype=x_type.dtype,
                shape=x_type.shape)


# ================================= Links ======================================

class ty_ChainerLinear():
    def __call__(self, linear, ty_args, ty_kwargs):
        x_type = ty_args[0]
        n_batch_axes, is_dummy_n_batch_axes = \
                get_kwarg(ty_kwargs, 'n_batch_axes', default=1)

        if linear.b is not None:
            assert x_type.dtype == linear.b.dtype
        if x_type.shape is None or is_dummy_n_batch_axes:
            return TyChainerVariable(x_type.dtype, shape=None)

        out_shape = self.infer_return_shape(linear, x_type.shape, n_batch_axes)
        return TyChainerVariable(x_type.dtype, shape=out_shape)

    def infer_return_shape(self, linear, x_shape, n_batch_axes):
        assert n_batch_axes >= 1

        if n_batch_axes > 1:
            batch_shape = x_shape[:n_batch_axes]
            batch_size = size_of_shape(batch_shape)
            x_shape = calculate_reshape(x_shape, (batch_size, -1))
        elif len(x_shape) > 2:
            x_shape = calculate_reshape(x_shape, (x_shape[0], -1))

        if linear.in_size is not None:
            assert x_shape[1] == linear.in_size

        y_shape = wrap_shape((x_shape[0], linear.out_size))
        if n_batch_axes > 1:
            y_shape = calculate_reshape(y_shape, (batch_shape + (-1,)))
        return y_shape


class ty_ChainerConvolution2D():
    def __call__(self, conv, ty_args, ty_kwargs):
        x_type, = ty_args

        assert x_type.dtype.kind == 'f'
        assert x_type.ndim == 4

        if is_incomplete_shape(x_type.shape):
            return TyChainerVariable(x_type.dtype, ndim=4)

        if conv.in_channels is not None:
            assert x_type.shape[1] == conv.in_channels

        return self.infer_return(conv, x_type)

        # TODO

        # w_type = type_of_value(conv.W)
        # b_type = type_of_value(conv.b)
        # self.groups = conv.groups

        # self.check_type_forward(make_multiple_tc_variable(
        #     [x_type, w_type, b_type], ('x', 'w', 'b')))

    def infer_return(self, conv, x_type):
        ksize = make_pair(conv.ksize)
        stride = make_pair(conv.stride)
        pad = make_pair(conv.pad)
        dilate = make_pair(conv.dilate)

        shape_2 = get_conv_outsize(
                x_type.shape[2], ksize[0], stride[0], pad[0], d=dilate[0])
        shape_3 = get_conv_outsize(
                x_type.shape[3], ksize[1], stride[1], pad[1], d=dilate[1])
        ret_shape = (x_type.shape[0], conv.out_channels, shape_2, shape_3)
        return TyChainerVariable(x_type.dtype, shape=wrap_shape(ret_shape))


class ty_ChainerBatchNormalization():
    def __call__(self, obj, ty_args, ty_kwargs):
        assert False


class ty_ChainerEmbedID():
    def __call__(self, embed, ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TyTensor)
        x_type = ty_args[0]
        w_type = embed.W

        if x_type.shape is None:
            return TyChainerVariable(w_type.dtype, shape=())  # TODO

        assert x_type.dtype.kind == 'i'
        assert len(x_type.shape) >= 1

        assert all([t < w_type.shape[0] for t in x_type.shape])
        out_type = x_type.shape + (w_type.shape[1],)
        return TyChainerVariable(w_type.dtype, shape=out_shape)


class ty_ChainerNStepBiLSTM():
    def __call__(self, nblstm, ty_args, ty_kwargs):
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
            hx_shape = wrap_shape(
                    (nblstm.n_layers * 2, len(xs_type.get_tys()), nblstm.out_size))
            hx_dtype = xs_type.get_tys()[0].dtype

        if isinstance(cx_type, TyTensor):
            cx_shape = cx_type.shape
            cx_dtype = cx_type.dtype
        else:
            cx_shape = wrap_shape(
                    (nblstm.n_layers * 2, len(xs_type.get_tys()), nblstm.out_size))
            cx_dtype = hx_dtype

        if hx_shape is None or cx_shape is None:
            # TODO: 適当
            return TyTuple([hx_type, cx_type, xs_type])

        assert hx_shape[0] // 2 == nblstm.n_layers
        assert hx_shape == cx_shape
        N = hx_shape[2]

        hy_type = TyChainerVariable(hx_dtype, shape=hx_shape)
        cy_type = TyChainerVariable(cx_dtype, shape=cx_shape)

        if any([t.shape is None for t in xs_type.get_tys()]):
            return TyTuple([hy_type, cy_type, TyList(TyChainerVariable(xs_dtypes[0]))])

        # TODO(momohatt): nblstm doesn't have attribute in_size
        # assert all([i == nblstm.in_size for _, i in xs_shapes])
        ys_shapes = [wrap_shape((l, 2 * N)) for l, _ in xs_shapes]
        ys_type = TyList([TyChainerVariable(xs_dtypes[0], shape=s) for s in ys_shapes])

        return TyTuple([hy_type, cy_type, ys_type])


ext_func_ty = {
        np.array                       : ty_NumpyArray(),
        np.cumsum                      : ty_ChainerIdentical(is_float_only=False),
        np.full                        : ty_NumpyFull(),
        np.ones                        : ty_NumpyOnes(),
        np.zeros                       : ty_NumpyOnes(),
        chainer.Variable               : ty_ChainerVariable(),
        cuda.to_cpu                    : ty_ChainerIdentical(is_float_only=False),
        F.average_pooling_2d           : ty_ChainerPooling2d(),
        F.broadcast_to                 : ty_ChainerBroadcastTo(),
        F.concat                       : ty_ChainerConcat(),
        F.dropout                      : ty_ChainerIdentical(),
        F.expand_dims                  : ty_ChainerExpandDims(),
        F.hstack                       : ty_ChainerHstack(),
        F.local_response_normalization : ty_ChainerLocalResponseNormalization(),
        F.max_pooling_2d               : ty_ChainerPooling2d(),
        F.pad                          : ty_ChainerPad(),
        F.pad_sequence                 : ty_ChainerPadSequence(),
        F.relu                         : ty_ChainerIdentical(),
        F.reshape                      : ty_ChainerReshape(),
        F.repeat                       : ty_ChainerRepeat(),
        F.separate                     : ty_ChainerSeparate(),
        F.sigmoid                      : ty_ChainerIdentical(),
        F.split_axis                   : ty_ChainerSplitAxis(),
        F.squeeze                      : ty_ChainerSqueeze(),
        F.softmax                      : ty_ChainerIdentical(),
        F.softmax_cross_entropy        : ty_ChainerSoftmaxCrossEntropy(),
        F.stack                        : ty_ChainerStack(),
        F.sum                          : ty_ChainerSum(),
        F.swapaxes                     : ty_ChainerSwapAxes(),
        F.tanh                         : ty_ChainerIdentical(),
        F.vstack                       : ty_ChainerVstack(),
        }


ext_callable_ty = {
        L.Linear             : ty_ChainerLinear(),
        L.Convolution2D      : ty_ChainerConvolution2D(),
        L.BatchNormalization : ty_ChainerBatchNormalization(),
        L.EmbedID            : ty_ChainerEmbedID(),
        L.NStepBiLSTM        : ty_ChainerNStepBiLSTM(),
        }
