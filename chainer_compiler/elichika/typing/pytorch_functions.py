import torch
import torch.nn as nn
import torch.nn.functional as F

from   chainer.utils import type_check

from   chainer_compiler.elichika.typing.ext_functions_utils import *
from   chainer_compiler.elichika.typing.types               import *
from   chainer_compiler.elichika.typing.pytorch.nn          import *
from   chainer_compiler.elichika.typing.pytorch.tensor      import *


class ty_TorchSequential():
    def nn(self, obj, ty_args, ty_kwargs):
        x_type, = ty_args
        for idx, module in enumerate(obj.modules()):
            if idx == 0: continue
            logic = pytorch_callable_ty[type(module)]
            x_type = logic(module, [x_type], {})
        return x_type


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


class ty_ChainerEmbedID():
    def __call__(self, embed, ty_args, ty_kwargs):
        assert isinstance(ty_args[0], TyTensor)
        x_type, = ty_args

        assert x_type.dtype.kind == 'i'
        assert x_type.ndim >= 1
        ret_shape = x_type.shape + (ShapeElem(embed.W.shape[1]),)

        if not is_incomplete_shape(x_type.shape):
            assert all([t < embed.W.shape[0] for t in x_type.shape])
        return TyTorchTensor(embed.W.dtype, shape=ret_shape)


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
        torch.reshape : ty_TorchReshape(),

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

        # https://pytorch.org/docs/stable/nn.functional.html#vision-functions
        F.interpolate : ty_TorchInterpolate(),

        torch.Tensor.add  : ty_TorchArith(torch.add),
        torch.Tensor.add_ : ty_TorchArith(torch.add),
        torch.Tensor.sub  : ty_TorchArith(torch.sub),
        torch.Tensor.sub_ : ty_TorchArith(torch.sub),
        torch.Tensor.mul  : ty_TorchArith(torch.mul),
        torch.Tensor.mul_ : ty_TorchArith(torch.mul),

        torch.Tensor.chunk     : ty_TorchChunk(),
        torch.Tensor.repeat    : ty_TorchRepeat(),
        torch.Tensor.squeeze   : ty_TorchSqueeze(),
        torch.Tensor.unsqueeze : ty_TorchUnsqueeze(),
        torch.Tensor.view      : ty_TorchView(),
        }


pytorch_callable_ty = {
        # https://pytorch.org/docs/stable/nn.html#containers
        nn.Sequential        : ty_TorchSequential().nn,

        # https://pytorch.org/docs/stable/nn.html#convolution-layers
        nn.Conv1d            : ty_TorchConv(dim=1).nn,
        nn.Conv2d            : ty_TorchConv(dim=2).nn,
        nn.Conv3d            : ty_TorchConv(dim=3).nn,
        nn.ConvTranspose1d   : ty_TorchConv(dim=1, transpose=True).nn,
        nn.ConvTranspose2d   : ty_TorchConv(dim=2, transpose=True).nn,
        nn.ConvTranspose3d   : ty_TorchConv(dim=3, transpose=True).nn,

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

        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        nn.LeakyReLU        : ty_TorchIdentical().nn,
        nn.ReLU             : ty_TorchIdentical().nn,
        nn.Sigmoid          : ty_TorchIdentical().nn,
        nn.Tanh             : ty_TorchIdentical().nn,

        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-other

        # https://pytorch.org/docs/stable/nn.html#normalization-layers
        nn.BatchNorm1d      : ty_TorchBatchNorm(dim=1).nn,
        nn.BatchNorm2d      : ty_TorchBatchNorm(dim=2).nn,
        nn.BatchNorm3d      : ty_TorchBatchNorm(dim=3).nn,
        nn.InstanceNorm1d   : ty_TorchInstanceNorm(dim=1).nn,
        nn.InstanceNorm2d   : ty_TorchInstanceNorm(dim=2).nn,
        nn.InstanceNorm3d   : ty_TorchInstanceNorm(dim=3).nn,

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
        nn.CrossEntropyLoss : ty_TorchNNCrossEntropyLoss().nn,

        # https://pytorch.org/docs/stable/nn.html#vision-layers
        nn.PixelShuffle     : ty_TorchPixelShuffle().nn,
        }
