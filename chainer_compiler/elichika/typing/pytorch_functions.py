import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from   chainer.utils.conv import get_conv_outsize
from   chainer.utils import type_check

from   chainer_compiler.elichika.typing.ext_functions_utils import *
from   chainer_compiler.elichika.typing.types import *


def check_dtype(module, dtype):
    for m in module.parameters():
        assert torch_dtype_to_np_dtype(m.dtype) == dtype, \
                "dtype mismatch in {}".format(module.__name__)
        # Checking the first param is enough
        return


# TODO: Unify with NumpyArray
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


class ty_TorchTensorOfShape():
    def __call__(self, ty_args, ty_kwargs):
        for ty in ty_args:
            unify(ty, TyInt())

        # TODO: use global default dtype
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', np.dtype('float32'))
        assert not lacks_dtype

        shape = wrap_shape([extract_value_from_ty(ty) for ty in ty_args])
        return TyNdarray(dtype, shape=shape)


# ==============================================================================

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


class ty_TorchArith():
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, ty_args, ty_kwargs):
        x_type, y_type = ty_args
        x, y = generate_dummy_value(x_type), generate_dummy_value(y_type)

        try:
            ty_ret = type_of_value(self.fn(x, y))
        except Exception as e:
            ty_ret = handle_inference_error(e, op.__class__.__name__, node)

        if is_incomplete_shape(x_type.shape) or \
                is_incomplete_shape(y_type.shape):
            ty_ret.shape = (ShapeElem(None),) * ty_ret.ndim
        return ty_ret


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


class ty_TorchView():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        shape_type = ty_args[1:]
        assert isinstance(x_type, TyTensor)

        out_shape = wrap_shape([extract_value_from_ty(t) for t in shape_type])
        ret_shape = calculate_reshape(x_type.shape, out_shape)
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


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
        check_dtype(obj, x_type.dtype)

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


class ty_TorchPad():
    def __init__(self, dim, is_const=False):
        self.dim = dim
        self.is_const = is_const

    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        assert x_type.ndim == self.dim + 2

        if self.is_const:
            if type(obj.value) is int:
                assert x_type.dtype.kind == 'i'
            elif type(obj.value) is float:
                assert x_type.dtype.kind == 'f'

        padding = make_tuple(obj.padding, self.dim + 2)
        return self.infer_return(x_type, padding)

    def infer_return(self, x_type, padding):
        shape = list(x_type.shape)
        for i in range(self.dim):
            shape[i + 2] = shape[i + 2] + padding[- (2 * i + 1)] + \
                    padding[- (2 * i + 2)]
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


class ty_TorchCat():
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, TySequence)
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        if lacks_value(xs_type) or lacks_dim:
            x_type = xs_type.get()
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim)

        self.check_type_forward(type_check.Variable(xs_type, 'xs'))
        return self.infer_return(xs_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.make_variable(self.dim, 'dim'))

        type_check.expect(
            -in_types[0].ndim <= self.dim,
            self.dim < in_types[0].ndim
        )
        ndim = type_check.eval(in_types[0].ndim)
        dim = self.dim % ndim
        for i in range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in range(0, ndim):
                if d == dim:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def infer_return(self, xs_type):
        ret_shape = list(xs_type[0].shape)
        ret_shape[self.dim] = sum([x_type.shape[self.dim] for x_type in xs_type])
        return TyTorchTensor(dtype=xs_type[0].dtype, shape=ret_shape)


class ty_TorchStack():
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, ty_args, ty_kwargs):
        xs_type, = ty_args
        assert isinstance(xs_type, TySequence)

        if xs_type.is_fixed_len:
            for ty in xs_type.get_tys():
                unify(xs_type.get_ty(), ty)

        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        if lacks_dim:
            x_type = xs_type.get()
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim + 1)

        self.dim %= xs_type.get().ndim + 1
        return self.infer_return(xs_type)

    def infer_return(self, xs_type):
        ret_shape = list(xs_type.get().shape)
        ret_shape.insert(self.dim, ShapeElem(xs_type.size()))
        return TyTorchTensor(xs_type.get().dtype, shape=ret_shape)


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


class ty_TorchSqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', None)

        if is_incomplete_shape(x_type.shape):
            if lacks_dim or self.dim is None:
                assert False, "torch.squeeze: cannot guess ndim of return type"

        return self.infer_return(x_type)

    def infer_return(self, x_type):
        if self.dim is not None:
            ret_shape = remove_dims(x_type.shape, (self.dim,))
        else:
            ret_shape = [s for s in x_type.shape if s != 1]
        return TyTorchTensor(x_type.dtype, shape=ret_shape)


class ty_TorchUnsqueeze():
    def __call__(self, ty_args, ty_kwargs):
        x_type, dim_type = ty_args
        assert isinstance(dim_type, TyNum)
        dim = extract_value_from_ty(dim_type)
        if dim is None:
            return TyTorchTensor(x_type.dtype, ndim=x_type.ndim + 1)

        shape = list(x_type.shape)
        shape.insert(dim, 1)
        return TyTorchTensor(x_type.dtype, shape=shape)


class ty_ChainerSum():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=None)
        self.keepdims, lacks_keepdims = \
                get_kwarg(ty_kwargs, 'keepdims', default=False)

        if isinstance(self.dim, int):
            self.dim = (self.dim,)

        self.check_type_forward(make_multiple_tc_variable(ty_args, ('x',)))

        if self.dim is None:
            self.dim = tuple(range(x_type.ndim))

        return self.infer_return(x_type)

    def check_type_forward(self, in_types):
        type_check.expect(in_types[0].dtype.kind == 'f')

        if self.dim is None:
            return

        for dim in self.dim:
            if dim >= 0:
                type_check.expect(
                    dim < in_types[0].ndim,
                )
            else:
                type_check.expect(
                    -dim - 1 < in_types[0].ndim,
                )

    def infer_return(self, x_type):
        if self.keepdims:
            ret_shape = list(x_type.shape)
            for i in self.dim:
                ret_shape[i] = 1
            return TyChainerVariable(x_type.dtype, shape=ret_shape)

        ret_shape = remove_dims(x_type.shape, self.dim)
        return TyChainerVariable(x_type.dtype, shape=ret_shape)


class ty_TorchChunk():
    def __call__(self, ty_args, ty_kwargs):
        x_type, chunk_type = ty_args
        assert isinstance(chunk_type, TyNum)
        chunks = chunk_type.value

        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)
        assert not lacks_dim
        return self.infer_return(x_type, chunks)

    def infer_return(self, x_type, chunks):
        ret_shape = list(x_type.shape)
        if chunks is None:
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        # TODO(momohatt): Handle cases where dim is not divisible by chunks
        ret_shape[self.dim] = ret_shape[self.dim] // chunks
        return TyTuple([TyTorchTensor(x_type.dtype, shape=ret_shape)
            for _ in range(chunks)])



class ty_TorchSplit():
    def __call__(self, ty_args, ty_kwargs):
        x_type, split_size_or_sections_type = ty_args
        self.dim, lacks_dim = get_kwarg(ty_kwargs, 'dim', default=0)

        # TODO: Handle cases where lacks_dim = True

        if isinstance(split_size_or_sections_type, TyNum):
            size = split_size_or_sections_type.value
            assert size is None or size > 0
            return self.infer_return_size(x_type, size)

        sections_type = split_size_or_sections_type
        assert isinstance(sections_type, TySequence)
        return self.infer_return_sections(x_type, sections_type)

    def infer_return_size(self, x_type, size):
        if size is None:
            if self.dim is None:
                return TyTuple(TyTorchTensor(x_type.dtype, ndim=x_type.ndim))
            ret_shape = list(x_type.shape)
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        if x_type.shape[self.dim].is_null():
            pass

        n_split = math.ceil(x_type.shape[self.dim].get_value() / size)
        if x_type.shape[self.dim] % size != 0:
            ret_shapes = [list(x_type.shape) for _ in range(n_split)]
            for i in range(n_split - 1):
                ret_shapes[i][self.dim] = size
            ret_shapes[-1][self.dim] = x_type.shape[self.dim] % size
            return TyTuple(
                    [TyTorchTensor(x_type.dtype, shape=shape) for shape in ret_shapes])

        ret_shape = list(x_type.shape)
        ret_shape[self.dim] = size
        return TyTuple(
            [TyTorchTensor(x_type.dtype, shape=ret_shape)] * n_split)

    def infer_return_sections(self, x_type, sections_type):
        if not sections_type.is_fixed_len:
            ret_shape = list(x_type.shape)
            ret_shape[self.dim] = None
            return TyTuple(TyTorchTensor(x_type.dtype, shape=ret_shape))

        sections = extract_value_from_ty(sections_type)
        ret_shapes = [list(x_type.shape) for _ in sections]
        for i, n in enumerate(sections):
            ret_shapes[i][self.dim] = n
        return TyTuple([TyTorchTensor(x_type.dtype, shape=shape)
            for shape in ret_shapes])


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
        n = ShapeElem(xs_type.size())
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
        check_dtype(obj, x_type.dtype)
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

        check_dtype(obj, x_type.dtype)
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


class ty_TorchLSTMCell():
    def nn(self, obj, ty_args, ty_kwargs):
        input_size = obj.input_size
        hidden_size = obj.hidden_size
        input_type = ty_args[0]
        assert isinstance(ty_args[1], TySequence)
        assert ty_args[1].is_fixed_len
        h_0_type, c_0_type = ty_args[1].get_tys()

        batch = input_type.shape[0]
        assert input_type.shape[1] == input_size
        assert h_0_type.shape[0] == batch
        assert h_0_type.shape[1] == hidden_size
        assert c_0_type.shape[0] == batch
        assert c_0_type.shape[1] == hidden_size

        return TyTuple([copy_ty(h_0_type), copy_ty(c_0_type)])


class ty_ChainerNStepBiLSTM():
    def __call__(self, nblstm, ty_args, ty_kwargs):
        hx_type, cx_type, xs_type = ty_args
        assert isinstance(xs_type, TySequence)
        xs_len = ShapeElem(xs_type.size())

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
        # https://pytorch.org/docs/stable/torch.html#creation-ops
        torch.tensor  : ty_TorchTensor(),
        torch.zeros   : ty_TorchTensorOfShape(),
        torch.ones    : ty_TorchTensorOfShape(),
        torch.rand    : ty_TorchTensorOfShape(),
        torch.randn   : ty_TorchTensorOfShape(),

        # https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
        torch.cat     : ty_TorchCat(),
        torch.chunk   : ty_TorchChunk(),
        torch.split   : ty_TorchSplit(),
        torch.stack   : ty_TorchStack(),

        # https://pytorch.org/docs/stable/torch.html#random-sampling
        torch.rand_like  : ty_TorchIdentical(),
        torch.randn_like : ty_TorchIdentical(),

        # https://pytorch.org/docs/stable/torch.html#math-operations
        torch.abs     : ty_TorchIdentical(),
        torch.cos     : ty_TorchIdentical(),
        torch.cosh    : ty_TorchIdentical(),
        torch.exp     : ty_TorchIdentical(),
        torch.log     : ty_TorchIdentical(),
        torch.sigmoid : ty_TorchIdentical(),
        torch.sin     : ty_TorchIdentical(),
        torch.sinh    : ty_TorchIdentical(),
        torch.sqrt    : ty_TorchIdentical(),
        torch.tan     : ty_TorchIdentical(),
        torch.tanh    : ty_TorchIdentical(),

        torch.mul     : ty_TorchArith(torch.mul),

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

        torch.Tensor.add  : ty_TorchArith(torch.add),
        torch.Tensor.add_ : ty_TorchArith(torch.add),
        torch.Tensor.sub  : ty_TorchArith(torch.sub),
        torch.Tensor.sub_ : ty_TorchArith(torch.sub),
        torch.Tensor.mul  : ty_TorchArith(torch.mul),
        torch.Tensor.mul_ : ty_TorchArith(torch.mul),

        torch.Tensor.view      : ty_TorchView(),
        torch.Tensor.chunk     : ty_TorchChunk(),
        torch.Tensor.squeeze   : ty_TorchSqueeze(),
        torch.Tensor.unsqueeze : ty_TorchUnsqueeze(),
        }


pytorch_callable_ty = {
        # https://pytorch.org/docs/stable/nn.html#containers
        nn.Sequential       : ty_TorchSequential().nn,

        # https://pytorch.org/docs/stable/nn.html#convolution-layers
        nn.Conv2d           : ty_TorchConv(dim=2).nn,

        # https://pytorch.org/docs/stable/nn.html#pooling-layers
        nn.AvgPool1d         : ty_TorchPooling(dim=1).nn,
        nn.AvgPool2d         : ty_TorchPooling(dim=2).nn,
        nn.AvgPool3d         : ty_TorchPooling(dim=3).nn,
        nn.MaxPool1d         : ty_TorchPooling(dim=1).nn,
        nn.MaxPool2d         : ty_TorchPooling(dim=2).nn,
        nn.MaxPool3d         : ty_TorchPooling(dim=3).nn,
        nn.AdaptiveAvgPool1d : ty_TorchAdaptivePooling(dim=1).nn,
        nn.AdaptiveAvgPool2d : ty_TorchAdaptivePooling(dim=2).nn,
        nn.AdaptiveAvgPool3d : ty_TorchAdaptivePooling(dim=3).nn,

        # https://pytorch.org/docs/stable/nn.html#padding-layers
        nn.ReflectionPad1d   : ty_TorchPad(dim=1).nn,
        nn.ReflectionPad2d   : ty_TorchPad(dim=2).nn,
        nn.ReplicationPad1d  : ty_TorchPad(dim=1).nn,
        nn.ReplicationPad2d  : ty_TorchPad(dim=2).nn,
        nn.ReplicationPad3d  : ty_TorchPad(dim=3).nn,
        nn.ZeroPad2d         : ty_TorchPad(dim=2).nn,
        nn.ConstantPad1d     : ty_TorchPad(dim=1, is_const=True).nn,
        nn.ConstantPad2d     : ty_TorchPad(dim=2, is_const=True).nn,
        nn.ConstantPad3d     : ty_TorchPad(dim=3, is_const=True).nn,

        # https://pytorch.org/docs/stable/nn.html#padding-layers

        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        nn.ReLU             : ty_TorchIdentical().nn,

        # https://pytorch.org/docs/stable/nn.html#recurrent-layers
        nn.LSTMCell         : ty_TorchLSTMCell().nn,

        # https://pytorch.org/docs/stable/nn.html#linear-layers
        nn.Linear           : ty_TorchLinear().nn,

        # https://pytorch.org/docs/stable/nn.html#dropout-layers
        nn.Dropout          : ty_TorchIdentical().nn,
        nn.Dropout2d        : ty_TorchIdentical(ndim_min=1).nn,
        nn.Dropout3d        : ty_TorchIdentical(ndim_min=1).nn,
        nn.AlphaDropout     : ty_TorchIdentical().nn,

        # https://pytorch.org/docs/stable/nn.html#loss-functions
        nn.CrossEntropyLoss : ty_TorchNNCrossEntropyLoss(),
        }
