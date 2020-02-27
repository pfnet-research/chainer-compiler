import numpy as np
import math

from   chainer_compiler.elichika.typing.ext.utils import *
from   chainer_compiler.elichika.typing.types     import *


class ty_NumpyAstype():
    def __call__(self, ty_args, ty_kwargs):
        x_type, dtype_type = ty_args
        if isinstance(dtype_type, TyString):
            return TyNdarray(np.dtype(dtype_type.value), shape=x_type.shape)
        print(dtype_type.t)
        return TyNdarray(dtype_type.t, shape=x_type.shape)


class ty_NumpyArray():
    def __call__(self, ty_args, ty_kwargs):
        x_type, = ty_args
        default_dtype = self.get_element_dtype(x_type)
        dtype, lacks_dtype = get_kwarg(ty_kwargs, 'dtype', default_dtype)
        assert not lacks_dtype, "numpy.array: dtype couldn't inferred"

        return TyNdarray(dtype, shape=self.calculate_shape(x_type))

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


class ty_NumpyIdentical():
    def __call__(self, ty_args, ty_kwargs):
        x_type = ty_args[0]
        assert isinstance(x_type, TyTensor)
        return copy_ty(x_type)


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

        assert isinstance(shape_type, (TyNum, TyTuple))

        shape = extract_value_from_ty(shape_type)
        if not isinstance(shape_type, TySequence):
            shape = (shape,)
        return TyNdarray(dtype, shape=shape)


numpy_func_ty = {
        np.ndarray.astype : ty_NumpyAstype(),
        np.array          : ty_NumpyArray(),
        np.cumsum         : ty_NumpyIdentical(),
        np.full           : ty_NumpyFull(),
        np.ones           : ty_NumpyOnes(),
        np.zeros          : ty_NumpyOnes(),
        }
