import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from   chainer.utils.conv import get_conv_outsize
from   chainer.utils import type_check

from   chainer_compiler.elichika.typing.types import *
from   chainer_compiler.elichika.typing.shape_elem import is_incomplete_shape

def size_of_shape(shape):
    size = 1
    for i in shape:
        size *= i
    return size


def make_tuple(x, n):
    if isinstance(x, tuple):
        return x
    return (x,) * n

def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    return x


def get_kwarg(ty_kwargs, key, default):
    if key in ty_kwargs.keys():
        # when unable to get the correct value, returns None
        return extract_value_from_ty(ty_kwargs[key]), lacks_value(ty_kwargs[key])
    return default, False


def extract_kwarg(ty_kwargs, key, default):
    if key in ty_kwargs.keys():
        return extract_kwarg(ty_kwargs[key])
    return default


def make_multiple_tc_variable(ty_args, names):
    assert len(ty_args) == len(names)
    return [type_check.Variable(t, n) for t, n in zip(ty_args, names)]


def calculate_reshape(orig_shape, input_shape):
    # orig_shape must be wrapped
    if is_incomplete_shape(orig_shape):
        if any([i == -1 for i in input_shape]):
            return wrap_shape([i if i != -1 else None for i in input_shape])
        return input_shape
    orig_shape = unwrap_shape(orig_shape)
    fill = abs(size_of_shape(orig_shape) // size_of_shape(input_shape))
    ret_shape = tuple([i if i != -1 else fill for i in input_shape])
    assert size_of_shape(orig_shape) == size_of_shape(ret_shape)
    return wrap_shape(ret_shape)


def remove_dims(shape, dims_to_remove):
    # dims_to_remove can have negative indices
    dims_to_remove = [d % len(shape) for d in dims_to_remove]
    return tuple([shape[i] for i in range(len(shape)) if i not in dims_to_remove])


class ty_TorchTensor():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        default_dtype = self.get_element_dtype(x_type)
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', default_dtype)
        assert not lacks_dtype, "torch.tensor: dtype couldn't inferred"

        return TyNdarray(dtype,
                shape=self.calculate_shape(x_type))

    def calculate_shape(self, x_type):
        if not isinstance(x_type, TySequence):
            return ()
        if not x_type.is_fixed_len:
            return (None,)

        x_tys = x_type.get_tys()

        if isinstance(x_tys[0], TySequence):
            list_lengths = [t.size() if t.is_fixed_len else None for t in x_tys]
            list_lengths_nonnull = [l for l in list_lengths if l is not None]

            # for example, we will not accept np.array([[1,2,3], [4,5]])
            assert all_same(list_lengths_nonnull), \
                    "numpy.array: incompatible list length"

        return (len(x_tys),) + self.calculate_shape(x_tys[0])

    def get_element_dtype(self, ty):
        # get element dtype of nested TySequence
        if isinstance(ty, TySequence):
            return self.get_element_dtype(ty.get())
        return tyobj2dtype(ty)


class ty_NumpyOnes():
    def __call__(self, ty_args, ty_kwargs):
        shape_type, = ty_args
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', np.dtype('float64'))

        assert not lacks_dtype

        if isinstance(shape_type, TyNum):
            assert shape_type.is_int()
        else:
            assert shape_type.is_fixed_len

        shape = extract_value_from_ty(shape_type)
        if isinstance(shape, int):
            shape = (shape,)

        return TyNdarray(dtype, shape=shape)


class ty_NumpyFull():
    def __call__(self, ty_args, ty_kwargs):
        shape_type, value_type = ty_args
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', tyobj2dtype(value_type))

        assert not lacks_dtype

        assert isinstance(shape_type, TyNum) or isinstance(shape_type, TyTuple)

        shape = extract_value_from_ty(shape_type)
        if not isinstance(shape_type, TySequence):
            shape = (shape,)
        return TyNdarray(dtype, shape=shape)

# ==============================================================================

class ty_TorchFlatten():
    def __call__(self, ty_args, ty_kwargs):
        input_type, = ty_args
        shape = input_type.shape
        start_dim, _ = get_kwarg(ty_kwargs, 'start_dim', default=0)
        end_dim, _   = get_kwarg(ty_kwargs, 'end_dim', default=-1)

        prefix_shape = shape[:start_dim]
        middle_shape = shape[start_dim:end_dim] + (shape[end_dim],)
        postfix_shape = shape[end_dim:][2:]
        size = size_of_shape(middle_shape)
        out_shape = prefix_shape + (size,) + postfix_shape
        return TyTorchTensor(shape=out_shape, dtype=input_type.dtype)


class ty_TorchPooling():
    def __init__(self, dim):
        self.dim = dim

    # TOOD(momohatt): in_channels, out_channels
    def __call__(self, ty_args, ty_kwargs):
        x_type, kernel_size_type = ty_args
        assert x_type.ndim == self.dim + 2
        assert x_type.dtype.kind == 'f'

        kernel_size = extract_value_from_ty(kernel_size_type)
        stride, _    = get_kwarg(ty_kwargs, 'stride', default=kernel_size)
        padding, _   = get_kwarg(ty_kwargs, 'padding', default=0)
        ceil_mode, _ = get_kwarg(ty_kwargs, 'ceil_mode', default=False)
        return self.infer_return(x_type, kernel_size, stride, padding, ceil_mode)

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.ndim == self.dim + 2
        assert x_type.dtype.kind == 'f'

        kernel_size = obj.kernel_size
        stride = obj.stride
        padding = obj.padding
        ceil_mode = obj.ceil_mode
        return self.infer_return(x_type, kernel_size, stride, padding, ceil_mode)

    def infer_return(self, x_type, kernel_size, stride, padding, ceil_mode):
        padding = make_tuple(padding, self.dim)
        kernel_size = make_tuple(kernel_size, self.dim)
        stride = make_tuple(stride, self.dim)
        shape = [0] * (self.dim + 2)

        shape[0] = x_type.shape[0]
        shape[1] = x_type.shape[1]
        if ceil_mode:
            for i in range(self.dim):
                shape[i + 2] = math.ceil((x_type.shape[i + 2] + padding[i] * 2 - kernel_size[i]) / stride[i]) + 1
        else:
            for i in range(self.dim):
                shape[i + 2] = (x_type.shape[i + 2] + padding[i] * 2 - kernel_size[i]) // stride[i] + 1

        return TyTorchTensor(x_type.dtype, shape=tuple(shape))


class ty_TorchAdaptivePooling():
    def __init__(self, dim):
        self.dim = dim

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        output_size = obj.output_size
        shape = x_type.shape[:-self.dim] + wrap_shape(output_size)
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_TorchNNCrossEntropyLoss():
    def nn(self, _, ty_args, ty_kwargs):
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
        return TyTorchTensor(x_type.dtype, shape=())


class ty_TorchIdentical():
    def __init__(self, ndim_min=None):
        self.ndim_min = ndim_min

    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        assert x_type.dtype.kind == 'f'
        if self.ndim_min:
            assert x_type.ndim >= self.ndim_min
        return copy_ty(x_type)

    def nn(self, _, ty_args, ty_kwargs):
        return self(ty_args, ty_kwargs)


# ========================= chainer.functions.array ============================

class ty_ChainerConcat():
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args

        assert isinstance(xs_type, TySequence)
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', default=1)

        if lacks_value(xs_type) or lacks_axis:
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
        for i in range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in range(0, ndim):
                if d == axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        ret_shape = list(xs_type[0].shape)
        ret_shape[self.axis] = sum([x_type.shape[self.axis] for x_type in xs_type])
        return TyChainerVariable(dtype=xs_type[0].dtype, shape=ret_shape)


class ty_ChainerStack():
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args

        assert isinstance(xs_type, TySequence)
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', default=1)

        if lacks_value(xs_type) or lacks_axis:
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
        for i in range(1, type_check.eval(in_types.size())):
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
        return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)


class ty_ChainerHstack():
    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args

        assert isinstance(xs_type, TySequence)
        if not xs_type.is_fixed_len:
            x_type = xs_type.get()
            if x_type.ndim < 2:
                ret_shape = (None,)
            else:
                ret_shape = (x_type.shape[0], None)
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check._argname((in_types[0],), ('x0',))

        ndim = type_check.eval(in_types[0].ndim)
        for i in range(1, type_check.eval(in_types.size())):
            type_check._argname((in_types[i],), ('x{}'.format(i),))
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                continue
            for d in range(0, ndim):
                if d == 1:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        if xs_type[0].ndim <= 1:
            return ty_ChainerConcat(axis=0).infer_return(xs_type)
        return ty_ChainerConcat(axis=1).infer_return(xs_type)


class ty_ChainerVstack():
    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args

        assert isinstance(xs_type, TySequence)
        if not xs_type.is_fixed_len:
            x_type = xs_type.get()
            if x_type.ndim < 2:
                ret_shape = (None, x_type.shape[0])
            else:
                ret_shape = (None, x_type.shape[1])
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

        ndim = type_check.eval(in_types[0].ndim)
        for i in range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            if ndim <= 1:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in range(1, ndim):
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        if xs_type[0].ndim <= 1:
            return ty_ChainerStack(axis=0).infer_return(xs_type)
        return ty_ChainerConcat(axis=0).infer_return(xs_type)


class ty_ChainerExpandDims():
    def __call__(self, ty_args, ty_kwargs):
        x_type, axis_type = ty_args
        self.axis = extract_value_from_ty(axis_type)

        if self.axis is None:
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim + 1)

        self.check_type_forward(make_multiple_tc_variable((x_type,), ('x',)))
        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        x_type, = in_types
        if self.axis >= 0:
            type_check.expect(x_type.ndim >= self.axis)
        else:
            type_check.expect(x_type.ndim >= -self.axis - 1)

    def infer_return(self, x_type):
        if self.axis < 0:
            self.axis = x_type.ndim + 1 - abs(self.axis)
        ret_shape = list(x_type.shape)
        ret_shape.insert(self.axis, 1)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerBroadcastTo():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_type = ty_args

        assert shape_type.is_fixed_len

        out_shape = wrap_shape(extract_value_from_ty(shape_type))

        # TODO: use check_type_forward
        ndim = len(out_shape)
        assert x_type.ndim <= ndim

        for i in range(-1, - x_type.ndim - 1, -1):
            assert x_type.shape[i] == out_shape[i] or x_type.shape[i] == 1

        return TyChainerVariable(x_type.dtype, shape=out_shape)


class ty_ChainerReshape():
    def __call__(self, ty_args, ty_kwargs):
        x_type, shape_type = ty_args

        assert shape_type.is_fixed_len

        self.shape = extract_value_from_ty(shape_type)
        return self.infer_return(x_type)

    def infer_return(self, x_type):
        ret_shape = calculate_reshape(x_type.shape, self.shape)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerRepeat():
    def __init__(self):
        self.axis = None

    def __call__(self, ty_args, ty_kwargs):
        x_type, repeats_type = ty_args
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', 0)

        if lacks_value(repeats_type) or lacks_axis:
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        repeats = extract_value_from_ty(repeats_type)
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
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args

        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', None)
        if isinstance(self.axis, int):
            self.axis = (self.axis,)

        if is_incomplete_shape(x_type.shape):
            # TODO: use ty_kwargs['axis'].size()
            if lacks_axis or self.axis is None:
                assert False, "chainer.fucntions.squeeze: cannot guess ndim of return type"

        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x',)))

        if self.axis is not None:
            for i in self.axis:
                assert x_type.shape[i] == 1, "chainer.fucntions.squeeze: invalid axis"
        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        # type_check.expect(in_types.size() == 1)
        x_type = in_types[0]

        if self.axis is not None:
            for x in self.axis:
                if x >= 0:
                    type_check.expect(x < x_type.ndim)
                else:
                    type_check.expect(-x_type.ndim <= x)

    def infer_return(self, x_type):
        if isinstance(self.axis, tuple):
            ret_shape = remove_dims(x_type.shape, self.axis)
        else:
            ret_shape = [s for s in x_type.shape if s != 1]
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSum():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', default=None)
        self.keepdims, lacks_keepdims = \
                get_kwarg(ty_kwargs, 'keepdims', default=False)

        if isinstance(self.axis, int):
            self.axis = (self.axis,)

        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x',)))

        if self.axis is None:
            self.axis = tuple(range(x_type.ndim))

        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types[0].dtype.kind == 'f')

        if self.axis is None:
            return

        for axis in self.axis:
            if axis >= 0:
                type_check.expect(
                    axis < in_types[0].ndim,
                )
            else:
                type_check.expect(
                    -axis - 1 < in_types[0].ndim,
                )

    def infer_return(self, x_type):
        if self.keepdims:
            ret_shape = list(x_type.shape)
            for i in self.axis:
                ret_shape[i] = 1
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        ret_shape = remove_dims(x_type.shape, self.axis)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSwapAxes():
    def __call__(self, ty_args, ty_kwargs):
        x_type, axis1_type, axis2_type = ty_args

        if lacks_value(axis1_type) or lacks_value(axis2_type):
            return TyChainerVariable(x_type.dtype, ndim=x_type.ndim)

        self.axis1 = extract_value_from_ty(axis1_type)
        self.axis2 = extract_value_from_ty(axis2_type)

        self.check_type_forward(type_check.make_variable(x_type, 'x'))
        return self.infer_return(x_type)

    def check_type_forward(self, x_type):
        type_check.expect(
                self.axis1 < x_type.ndim,
                self.axis2 < x_type.ndim
                )

    def infer_return(self, x_type):
        ret_shape = list(x_type.shape)
        ret_shape[self.axis1], ret_shape[self.axis2] = \
                ret_shape[self.axis2], ret_shape[self.axis1]
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerSeparate():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.axis, lacks_axis = get_kwarg(ty_kwargs, 'axis', 0)

        if lacks_axis:
            return TyTuple(TyChainerVariable(x_type.dtype, ndim=x_type.ndim-1))

        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x',)))
        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        x_type = in_types[0]
        if self.axis >= 0:
            type_check.expect(self.axis < x_type.ndim)
        else:
            type_check.expect(-self.axis <= x_type.ndim)

    def infer_return(self, x_type):
        n = x_type.shape[self.axis]
        ret_shape = x_type.shape[:self.axis] + x_type.shape[self.axis + 1:]
        ret_ty = TyChainerVariable(x_type.dtype, shape=ret_shape)
        if not n.has_value():
            return TyTuple(ret_ty)
        return TyTuple([ret_ty] * n.value)


class ty_ChainerSplitAxis():
    def __call__(self, ty_args, ty_kwargs):
        x_type, _, axis_type = ty_args

        self.axis = axis_type.value

        if isinstance(ty_args[1], TyNum):
            sections = ty_args[1].value
            return self.infer_return(x_type, sections, is_indices=False)

        # 1-D array
        indices_type = ty_args[1]
        assert isinstance(indices_type, TyTensor)

        assert indices_type.ndim == 1
        n = indices_type.shape[0].value
        return self.infer_return(x_type, n + 1, is_indices=True)

    # TODO: check_type_forward

    def infer_return(self, x_type, n_split, is_indices):
        if n_split is None:
            if self.axis is None:
                return TyTuple(TyChainerVariable(x_type.dtype, ndim=x_type.ndim))
            ret_shape = list(x_type.shape)
            ret_shape[self.axis] = None
            return TyTuple(TyChainerVariable(x_type.dtype, shape=ret_shape))
        ret_shape = list(x_type.shape)
        if is_indices:
            ret_shape[self.axis] = None
        else:
            ret_shape[self.axis] = ret_shape[self.axis] // n_split
        return TyTuple(
            [TyChainerVariable(x_type.dtype, shape=ret_shape)] * n_split)


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

        pad_width = extract_value_from_ty(pad_width_type)
        if isinstance(pad_width, int):
            pad_width = make_pair(pad_width)
        if isinstance(pad_width[0], int):
            pad_width = pad_width * x_type.ndim
        for pad in pad_width:
            assert len(pad) == 2, "chainer.functions.pad: pad_width is invalid"
        return self.infer_return(x_type, pad_width)

    def check_type_forward(self, in_types):
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

    def infer_return(self, x_type, pad_width):
        ret_shape = list(x_type.shape)
        for i in range(x_type.ndim):
            ret_shape[i] += pad_width[i][0] + pad_width[i][1]
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_ChainerPadSequence():
    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        self.length, lacks_length = get_kwarg(ty_kwargs, 'length', None)

        if not xs_type.is_fixed_len:
            ret_shape = list((None,) * (xs_type.get().ndim + 1))
            if not lacks_length:
                ret_shape[1] = self.length
            return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)

        self.check_type_forward(type_check.make_variable(xs_type, 'xs'))
        return self.infer_return(xs_type, lacks_length)

    def check_type_forward(self, xs_type):
        for i in range(xs_type.size().eval()):
            type_check.expect(
                xs_type[i].ndim > 0,
                xs_type[i].shape[1:] == xs_type[0].shape[1:],
                xs_type[i].dtype == xs_type[0].dtype
            )

    def infer_return(self, xs_type, lacks_length):
        n = len(xs_type)
        ret_shape = list((n,) + xs_type.get().shape)

        if lacks_length:
            ret_shape[1] = None
            return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)
        if self.length is not None:
            ret_shape[1] = self.length
            return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)

        shape_0s = [t.shape[0] for t in xs_type.get_tys()]
        ret_shape[1] = max(shape_0s)
        return TyChainerVariable(xs_type.get().dtype, shape=ret_shape)


class ty_ChainerLocalResponseNormalization():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args

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
        return TyChainerVariable(dtype=x_type.dtype, shape=x_type.shape)


class ty_TorchLinear():
    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.dtype == np.dtype('float32')
        assert x_type.shape[-1] == obj.in_features

        return self.infer_return_shape(x_type, obj.out_features)

    def infer_return_shape(self, x_type, out_features):
        out_shape = x_type.shape[:-1] + (out_features,)
        return TyTorchTensor(x_type.dtype, shape=out_shape)


class ty_TorchConv():
    def __init__(self, dim):
        self.dim = dim

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args

        assert x_type.dtype == np.dtype('float32')
        assert x_type.ndim == self.dim + 2
        assert x_type.shape[1] == obj.in_channels

        return self.infer_return(x_type, obj)

    def infer_return(self, x_type, obj):
        kernel_size = make_tuple(obj.kernel_size, self.dim)
        stride      = make_tuple(obj.stride,      self.dim)
        padding     = make_tuple(obj.padding,     self.dim)
        dilation    = make_tuple(obj.dilation,    self.dim)

        shape = [0] * self.dim
        for i in range(self.dim):
            shape[i] = get_conv_outsize(
                x_type.shape[i + 2], kernel_size[i], stride[i], padding[i], d=dilation[i])
        ret_shape = (x_type.shape[0], obj.out_channels) + tuple(shape)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_TorchSequential():
    def nn(self, seq, ty_args, ty_kwargs):
        x_type, = ty_args
        for idx, module in enumerate(seq.modules()):
            if idx == 0: continue
            print("---", module)
            logic = pytorch_callable_ty[type(module)]
            x_type = logic.nn(module, [x_type], {})
        return x_type


class ty_ChainerBatchNormalization():
    def __call__(self, obj, ty_args, ty_kwargs):
        assert False


class ty_ChainerEmbedID():
    def __call__(self, embed, ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TyTensor)
        x_type, = ty_args

        assert x_type.dtype.kind == 'i'
        assert x_type.ndim >= 1
        ret_shape = x_type.shape + (ShapeElem(embed.W.shape[1]),)

        if not is_incomplete_shape(x_type.shape):
            assert all([t < embed.W.shape[0] for t in x_type.shape])
        return TyChainerVariable(embed.W.dtype, shape=ret_shape)


class ty_ChainerNStepBiLSTM():
    def __call__(self, nblstm, ty_args, ty_kwargs):
        hx_type, cx_type, xs_type = ty_args
        assert isinstance(xs_type, TySequence)

        xs_len = len(xs_type.get_tys()) if xs_type.is_fixed_len else None

        if isinstance(hx_type, TyTensor):
            hx_shape = hx_type.shape
            hx_dtype = hx_type.dtype
        else:
            hx_shape = (nblstm.n_layers * 2, xs_len, nblstm.out_size)
            hx_dtype = xs_type.get().dtype

        if isinstance(cx_type, TyTensor):
            cx_shape = cx_type.shape
            cx_dtype = cx_type.dtype
        else:
            cx_shape = (nblstm.n_layers * 2, xs_len, nblstm.out_size)
            cx_dtype = hx_dtype

        hy_type = TyChainerVariable(hx_dtype, shape=hx_shape)
        cy_type = TyChainerVariable(cx_dtype, shape=cx_shape)

        assert hx_shape[0] // 2 == nblstm.n_layers
        assert hx_shape == cx_shape
        N = hx_shape[2]

        if not xs_type.is_fixed_len:
            # TODO
            ys_shape = (xs_type.get().shape[0], 2 * N)
            ys_type = TyList(TyChainerVariable(xs_type.get().dtype, shape=ys_shape))
            return TyTuple([hy_type, cy_type, ys_type])

        xs_dtypes = [t.dtype for t in xs_type.get_tys()]
        xs_shapes = [t.shape for t in xs_type.get_tys()]
        assert all_same(xs_dtypes)

        # TODO(momohatt): nblstm doesn't have attribute in_size
        # assert all([i == nblstm.in_size for _, i in xs_shapes])
        ys_shapes = [(l, 2 * N) for l, _ in xs_shapes]
        ys_type = TyList([TyChainerVariable(xs_dtypes[0], shape=s) for s in ys_shapes])

        return TyTuple([hy_type, cy_type, ys_type])


pytorch_func_ty = {
        np.full       : ty_NumpyFull(),
        np.ones       : ty_NumpyOnes(),
        np.zeros      : ty_NumpyOnes(),

        torch.tensor  : ty_TorchTensor(),
        torch.flatten : ty_TorchFlatten(),

        # https://pytorch.org/docs/stable/nn.functional.html#pooling-functions
        F.avg_pool1d  : ty_TorchPooling(dim=1),
        F.avg_pool2d  : ty_TorchPooling(dim=2),
        F.avg_pool3d  : ty_TorchPooling(dim=3),
        F.max_pool1d  : ty_TorchPooling(dim=1),
        F.max_pool2d  : ty_TorchPooling(dim=2),
        F.max_pool3d  : ty_TorchPooling(dim=3),

        # https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
        F.relu        : ty_TorchIdentical(),
        F.log_softmax : ty_TorchIdentical(),
        F.tanh        : ty_TorchIdentical(),
        F.sigmoid     : ty_TorchIdentical(),
        }


pytorch_callable_ty = {
        # https://pytorch.org/docs/stable/nn.html#containers
        nn.Sequential       : ty_TorchSequential(),

        # https://pytorch.org/docs/stable/nn.html#convolution-layers
        nn.Conv2d           : ty_TorchConv(dim=2),

        # https://pytorch.org/docs/stable/nn.html#pooling-layers
        nn.AvgPool1d         : ty_TorchPooling(dim=1),
        nn.AvgPool2d         : ty_TorchPooling(dim=2),
        nn.AvgPool3d         : ty_TorchPooling(dim=3),
        nn.MaxPool1d         : ty_TorchPooling(dim=1),
        nn.MaxPool2d         : ty_TorchPooling(dim=2),
        nn.MaxPool3d         : ty_TorchPooling(dim=3),
        nn.AdaptiveAvgPool1d : ty_TorchAdaptivePooling(dim=1),
        nn.AdaptiveAvgPool2d : ty_TorchAdaptivePooling(dim=2),
        nn.AdaptiveAvgPool3d : ty_TorchAdaptivePooling(dim=3),

        # https://pytorch.org/docs/stable/nn.html#padding-layers

        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        nn.ReLU             : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/nn.html#linear-layers
        nn.Linear           : ty_TorchLinear(),

        # https://pytorch.org/docs/stable/nn.html#dropout-layers
        nn.Dropout          : ty_TorchIdentical(),
        nn.Dropout2d        : ty_TorchIdentical(ndim_min=1),
        nn.Dropout3d        : ty_TorchIdentical(ndim_min=1),
        nn.AlphaDropout     : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/nn.html#loss-functions
        nn.CrossEntropyLoss : ty_TorchNNCrossEntropyLoss(),
        }
