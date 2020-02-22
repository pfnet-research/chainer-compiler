from   copy import deepcopy
from   enum import Enum, IntEnum

import chainer
import numpy as np

import torch

from   chainer_compiler.elichika.typing import utils
from   chainer_compiler.elichika.typing.shape_elem import ShapeElem, wrap_shape, unwrap_shape, unify_shape

def print_warning(msg):
    print("\x1b[33m[WARNING] " + msg + "\x1b[39m")

class TyObj():  # base type, meaning 'unknown'
    def __init__(self):
        self.is_optional = False
    def __str__(self):
        if self.is_optional:
            return "optional({})".format(self.show())
        return self.show()
    # TODO(momohatt): fix __repr__
    def __repr__(self):
        return self.__str__()

    def is_mutable(self):
        pass
    # dereference internal type
    def deref(self):
        return self

# --------------------------- python primivite types ---------------------------

class TyNone(TyObj):
    def show(self):
        return "NoneType"
    def __eq__(self, other):
        return isinstance(other, TyNone)
    def is_mutable(self):
        return False


class NumKind(IntEnum):
    BOOL = 0
    INT = 1
    FLOAT = 2

    def __str__(self):
        if self.value == 0:
            return "bool"
        if self.value == 1:
            return "int"
        if self.value == 2:
            return "float"


class TyNum(TyObj):
    def __init__(self, kind, value=None):
        super().__init__()
        self.kind = kind
        self.value = value

    def show(self):
        return str(NumKind(self.kind))

    def __eq__(self, other):
        return isinstance(other, TyNum) and self.kind == other.kind

    def is_mutable(self):
        return False

    def coerce_value(self):
        if self.value is None:
            return
        self.value = eval(str(NumKind(self.kind)))(self.value)

    def is_int(self):
        return self.kind <= NumKind.INT


def TyBool(value=None):
    return TyNum(0, value=value)  # bool or int or float

def TyInt(value=None):
    return TyNum(1, value=value)  # int or float

def TyFloat(value=None):
    return TyNum(2, value=value)  # float


class TyString(TyObj):
    def __init__(self, value=None):
        super().__init__()
        self.value = value
    def show(self):
        return "string"
    def __eq__(self, other):
        return isinstance(other, TyString)
    def is_mutable(self):
        return False


class TyArrow(TyObj):
    def __init__(self, argty, retty):
        super().__init__()
        self.argty = argty  # Arguments are uncurried
        self.retty = retty

    def show(self):
        if self.argty == []:
            return "(no argument) -> {}".format(self.retty)
        return "".join([str(t) + " -> " for t in self.argty]) + str(self.retty)

    def __eq__(self, other):
        return isinstance(other, TyArrow) and self.argty == other.argty and \
                self.retty == other.retty

    def is_mutable(self):
        return False

    def deref(self):
        self.argty = [t.deref() for t in self.argty]
        self.retty = self.retty.deref()
        return self


class SequenceKind(Enum):
    LIST = 0
    TUPLE = 1

class TySequence(TyObj):
    def __init__(self, ty, kind):
        super().__init__()
        self.kind = kind
        self.is_fixed_len = isinstance(ty, list)
        self._ty = ty

    def show(self):
        if self.is_fixed_len:
            if self.kind == SequenceKind.LIST:
                return "[" + utils.intercalate([str(t) for t in self._ty], ", ") + "]"

            if self.kind == SequenceKind.TUPLE:
                if len(self._ty) == 1:
                    return "(" + str(self._ty[0]) + ",)"
                return "(" + utils.intercalate([str(t) for t in self._ty], ", ") + ")"

            return "{" + utils.intercalate([str(t) for t in self._ty], ", ") + "}"

        if self.kind == SequenceKind.LIST:
            return str(self._ty) + " list"
        if self.kind == SequenceKind.TUPLE:
            return str(self._ty) + " tuple"

        return str(self._ty) + " sequence"

    def __eq__(self, other):
        return isinstance(other, TySequence) and self._ty == other._ty

    def __getitem__(self, i):
        assert self.is_fixed_len
        return self._ty[i]

    def __len__(self):
        assert self.is_fixed_len
        return len(self._ty)

    # def __iter__(self):
    #     assert self.is_fixed_len
    #     self._i = 0
    #     return self

    # def __next__(self):
    #     if self._i < len(self._ty): raise StopIteration
    #     self._i += 1
    #     return self._i - 1

    def is_mutable(self):
        return self.kind == SequenceKind.LIST

    def size(self):
        assert self.is_fixed_len
        return len(self._ty)

    def deref(self):
        if self.is_fixed_len is not None:
            if self.is_fixed_len:
                self._ty = [t.deref() for t in self._ty]
            else:
                self._ty = self._ty.deref()
        return self

    def get(self):
        # get one type as a representative
        if self.is_fixed_len:
            return self.get_tys()[0]
        return self.get_ty()

    def get_ty(self):
        assert not self.is_fixed_len
        return self._ty

    def get_tys(self):
        assert self.is_fixed_len
        return self._ty

    def coerce_to_variable_len(self, ty=None):
        # does nothing if self is not fixed-length
        if self.is_fixed_len:
            if ty is None:
                ty = TyVar()
            for t in self._ty:
                unify(ty, t, inspect_shape=False)
            self._ty = ty
            self.is_fixed_len = False
        return

    def is_list(self):
        return self.kind == SequenceKind.LIST

    def is_tuple(self):
        return self.kind == SequenceKind.TUPLE


def TyList(ty):  # shorthand notation
    return TySequence(ty, SequenceKind.LIST)

def TyTuple(ty):  # shorthand notation
    return TySequence(ty, SequenceKind.TUPLE)


class TyDict(TyObj):
    # TODO(momohatt): Support hetero-value dicts (simply set valty to 'TyObj',
    # or infer type of each fields (ideally))
    def __init__(self, keyty, valty):
        super().__init__()
        self.keyty = keyty
        self.valty = valty

    def show(self):
        return "{" + str(self.keyty) + " : " + str(self.valty) + "}"

    def __eq__(self, other):
        return isinstance(other, TyDict) and self.keyty == other.keyty and \
                self.valty == other.valty

    def is_mutable(self):
        return True

    def deref(self):
        self.keyty = self.keyty.deref()
        self.valty = self.valty.deref()
        return self


class TyUserDefinedClass(TyObj):
    def __init__(self, name, instance):
        super().__init__()
        self.name = name
        # XXX: we will assume that an instance already exists
        self.instance = instance

    def show(self):
        return "class " + self.name

    def is_mutable(self):
        return True


# --------------------- numpy ndarray / chainer variable -----------------------

class TensorKind(Enum):
    ndarray = 0
    chainer_variable = 1
    torch_tensor = 2


class TyDType(TyObj):
    def __init__(self, t=None):
        super().__init__()
        if t is None:
            self.t = None
        else:
            self.t = np.dtype(t)
    def __str__(self):
        return "dtype({})".format(str(self.t))
    def __eq__(self, other):
        return self.t == other.t


class TyTensor(TyObj):
    def __init__(self, dtype, kind, ndim, shape=None):  # we do not allow heterogeneous type ndarray
        super().__init__()
        if isinstance(dtype, torch.dtype):
            self.dtype = torch_dtype_to_np_dtype(dtype)
        else:
            self.dtype = np.dtype(dtype)
        self.kind = kind
        self.ndim = ndim
        if shape is None:
            shape = (None,) * ndim
        self.shape = wrap_shape(shape)  # Tuple[ShapeElem]

    def show(self):
        if self.kind == TensorKind.ndarray:
            return "ndarray(dtype={}, shape={})".format(self.dtype, self.shape)
        if self.kind == TensorKind.chainer_variable:
            return "Variable(dtype={}, shape={})".format(self.dtype, self.shape)
        if self.kind == TensorKind.torch_tensor:
            return "torch.Tensor(dtype={}, shape={})".format(self.dtype, self.shape)

    def __eq__(self, other):
        # TODO: shape?
        return isinstance(other, TyTensor) and self.dtype == other.dtype

    def is_mutable(self):
        return True

    def is_ndarray(self):
        return self.kind == TensorKind.ndarray

    def is_chainer_variable(self):
        return self.kind == TensorKind.chainer_variable

    def is_torch_tensor(self):
        return self.kind == TensorKind.torch_tensor


def TyNdarray(dtype, ndim=None, shape=None):
    # ndim and shape cannot be None at the same time
    if ndim is None:
        ndim = len(shape)
    return TyTensor(dtype, TensorKind.ndarray, ndim=ndim, shape=shape)

def TyChainerVariable(dtype, ndim=None, shape=None):
    if ndim is None:
        ndim = len(shape)
    return TyTensor(dtype, TensorKind.chainer_variable, ndim=ndim, shape=shape)

def TyTorchTensor(dtype, ndim=None, shape=None):
    if ndim is None:
        ndim = len(shape)
    return TyTensor(dtype, TensorKind.torch_tensor, ndim=ndim, shape=shape)

def torch_dtype_to_np_dtype(dtype):
    # TODO(momohatt): Better way to do this?
    dtype_dict = {
            torch.bool    : np.dtype(np.bool),
            torch.uint8   : np.dtype(np.uint8),
            torch.int8    : np.dtype(np.int8),
            torch.int16   : np.dtype(np.int16),
            torch.short   : np.dtype(np.int16),
            torch.int32   : np.dtype(np.int32),
            torch.int     : np.dtype(np.int32),
            torch.int64   : np.dtype(np.int64),
            torch.long    : np.dtype(np.int64),
            torch.float16 : np.dtype(np.float16),
            torch.half    : np.dtype(np.float16),
            torch.float32 : np.dtype(np.float32),
            torch.float   : np.dtype(np.float32),
            torch.float64 : np.dtype(np.float64),
            torch.double  : np.dtype(np.float64),
            }
    return dtype_dict[dtype]


# ---------------------- InferenceEngine internal types ------------------------

var_counter = 0

class TyVar(TyObj):
    def __init__(self, lineno=None):
        global var_counter
        super().__init__()
        self.i = var_counter
        var_counter += 1
        self.ty = None
        self.is_set = False
        self.lineno = lineno

    def show(self):
        if self.ty is not None:
            return str(self.ty)
        if self.lineno is not None:
            return "a{} (from line {})".format(self.i, self.lineno)
        return "a{}".format(self.i)

    def __eq__(self, other):
        return self.deref() == other.deref()

    def is_mutable(self):
        if self.is_set:
            return self.ty.is_mutable()
        return False

    def set(self, ty):
        assert self.is_set == False
        self.is_set = True
        self.ty = ty

    def deref(self):
        if self.is_set:
            return self.ty.deref()
        return self


# ------------------------------------------------------------------------------

def all_same_ty(tys):
    _tytmp = TyVar()
    for t in tys:
        unify(_tytmp, t)
    return True


def all_same(l):
    return all([e == l[0] for e in l])


def type_of_value(value):
    if value is None:
        return TyNone()
    if isinstance(value, bool):
        return TyBool(value=value)
    if isinstance(value, int):
        return TyInt(value=value)
    if isinstance(value, float):
        return TyFloat(value=value)
    if isinstance(value, str):
        return TyString(value=value)
    if isinstance(value, list):
        return TyList([type_of_value(v) for v in value])
    if isinstance(value, range):
        return TyList([type_of_value(v) for v in value])
    if isinstance(value, tuple):
        return TyTuple([type_of_value(v) for v in value])
    if isinstance(value, dict):
        if len(value) == 0:
            return TyDict(TyVar(), TyVar())
        return TyDict(type_of_value(list(value.keys())[0]),
                type_of_value(list(value.items())[0]))
    if isinstance(value, np.ndarray):
        return TyNdarray(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, chainer.Variable):
        return TyChainerVariable(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, torch.Tensor):
        return TyTorchTensor(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, np.dtype):
        return TyDType(value)
    if isinstance(value, type) and value in np.typeDict.values():
        # XXX: np.typeDict.values() is a list of all dtypes
        return TyDType(value)
    if isinstance(value, torch.dtype):
        return TyDType(torch_dtype_to_np_dtype(value))
    if isinstance(value, ShapeElem):
        if isinstance(value.value, int):
            return TyInt(value.value)
        return TyInt()

    return TyUserDefinedClass(type(value).__name__, value)


def lacks_value(ty) -> bool:
    ty = ty.deref()

    if isinstance(ty, TyNone):
        return False
    if isinstance(ty, TyNum):
        return ty.value is None
    if isinstance(ty, TyString):
        return ty.value is None
    if isinstance(ty, TySequence):
        if not ty.is_fixed_len:
            return True
        return any([lacks_value(t) for t in ty.get_tys()])
    if isinstance(ty, TyDict):
        return True
    if isinstance(ty, TyTensor):
        return ty.shape is None or any([not i.has_value() for i in ty.shape])
    if isinstance(ty, TyDType):
        return ty.t is None


def generate_dummy_value(ty) -> object:
    # creates dummy value

    ty = ty.deref()

    if isinstance(ty, TyNone):
        return None
    if isinstance(ty, TyNum):
        if ty.value is not None:
            return ty.value
        return eval(str(NumKind(ty.kind)))(1)  # XXX: use 1 to avoid division by zero
    if isinstance(ty, TyString):
        if ty.value is not None:
            return ty.value
        return ""
    if isinstance(ty, TySequence):
        if ty.is_fixed_len:
            ret = [generate_dummy_value(t) for t in ty.get_tys()]
        else:
            ret = [generate_dummy_value(ty.get_ty())]
        if ty.is_list():
            return ret
        return tuple(ret)
    if isinstance(ty, TyDict):
        return { generate_dummy_value(ty.keyty) : generate_dummy_value(ty.valty) }
    if isinstance(ty, TyTensor):
        ret = np.zeros(dtype=ty.dtype, shape=unwrap_shape(ty.shape))
        if ty.is_ndarray():
            return ret
        if ty.is_chainer_variable():
            return chainer.Variable(ret)
        if ty.is_torch_tensor():
            return torch.as_tensor(ret)
    if isinstance(ty, TyDType):
        return ty.t

    assert False, "generate_dummy_value: type not understood: " + str(ty)


def extract_value_from_ty(ty):
    # returns None where it doesn't have value
    ty = ty.deref()

    if isinstance(ty, TyNone):
        return None
    if isinstance(ty, TyNum):
        if ty.value is not None:
            return ty.value
        return None
    if isinstance(ty, TyString):
        if ty.value is not None:
            return ty.value
        return None
    if isinstance(ty, TySequence):
        if not ty.is_fixed_len:
            return None
        ret = [extract_value_from_ty(t) for t in ty.get_tys()]
        if ty.is_list():
            return ret
        return tuple(ret)
    if isinstance(ty, TyDict):
        return None
    if isinstance(ty, TyTensor):
        return None
    if isinstance(ty, TyDType):
        return ty.t

    assert False, "extract_value_from_ty: type not understood: " + str(ty)


def choose_stronger_ty(ty1, ty2):
    if isinstance(ty1, TyNone):
        return ty2
    if isinstance(ty2, TyNone):
        return ty1
    return ty1  # whichever is okay


def copy_ty(ty):
    if isinstance(ty, (TyNone, TyNum, TyString)):
        ret = deepcopy(ty)
    elif isinstance(ty, TyArrow):
        ret = TyArrow([copy_ty(t) for t in ty.argty], copy_ty(ty.retty))
    elif isinstance(ty, TySequence):
        if ty.is_fixed_len:
            ret = TySequence([copy_ty(t) for t in ty.get_tys()], ty.kind)
        else:
            ret = TySequence(copy_ty(ty.get_ty()), ty.kind)
    elif isinstance(ty, TyDict):
        # XXX: do not copy instance
        ret = TyDict(ty.keyty, ty.valty)
    elif isinstance(ty, TyUserDefinedClass):
        ret = TyUserDefinedClass(ty.name, ty.instance)
    elif isinstance(ty, TyDType):
        ret = TyDType()
    elif isinstance(ty, TyTensor):
        ret = TyTensor(ty.dtype, ty.kind, ty.ndim, shape=ty.shape)
    elif isinstance(ty, TyVar):
        ret = TyVar(None)
        if ty.ty is not None:
            ret.set(ty.deref())

    ret.is_optional = ty.is_optional
    return ret


def tyobj2dtype(ty):
    if isinstance(ty, TyNum):
        return np.dtype(str(NumKind(ty.kind)))
    assert False


# ==============================================================================

class UnifyError(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "UnifyError: {} and {} are not unifiable".format(ty1, ty2)


def occur(var, ty):
    if isinstance(ty, TyVar):
        if var is ty:
            return True
        occur(var, ty.ty)
    if isinstance(ty, TyArrow):
        return any([occur(var, t) for t in ty.argty]) or occur(var, ty.retty)
    if isinstance(ty, TySequence):
        if ty.is_fixed_len:
            return any([occur(var, t) for t in ty.get_tys()])
        return occur(var, ty.get_ty())
    if isinstance(ty, TyDict):
        return occur(var, ty.keyty) or occur(var, ty.valty)
    return False


def unify(ty1, ty2, inspect_shape=True):
    # inspect_shape: forces shapes to be identical iff True
    ty1 = ty1.deref()
    ty2 = ty2.deref()

    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return

    if isinstance(ty1, TyVar):
        if isinstance(ty2, TyVar) and ty1 is ty2:
            return
        if occur(ty1, ty2):
            raise UnifyError(ty1, ty2)
        ty1.set(ty2)
        return

    if isinstance(ty2, TyVar):
        if occur(ty2, ty1):
            raise UnifyError(ty1, ty2)
        ty2.set(ty1)
        return

    if isinstance(ty1, TyNone):
        ty2.is_optional = True
        return

    if isinstance(ty2, TyNone):
        ty1.is_optional = True
        return

    ty1.is_optional = ty2.is_optional = ty1.is_optional or ty2.is_optional

    if isinstance(ty1, TyNum) and isinstance(ty2, TyNum):
        ty1.kind = ty2.kind = max(ty1.kind, ty2.kind)
        ty1.coerce_value()
        ty2.coerce_value()
        return

    if isinstance(ty1, TyString) and isinstance(ty2, TyString):
        return

    if isinstance(ty1, TyArrow) and isinstance(ty2, TyArrow) and \
            len(ty1.argty) == len(ty2.argty):
        for (at1, at2) in zip(ty1.argty, ty2.argty):
            unify(at1, at2)
        unify(ty1.retty, ty2.retty)
        return

    if isinstance(ty1, TySequence) and isinstance(ty2, TySequence):
        if ty1.is_fixed_len and ty2.is_fixed_len:
            if not len(ty1.get_tys()) == len(ty2.get_tys()):
                ty1.coerce_to_variable_len()
                ty2.coerce_to_variable_len()
                unify(ty1.get_ty(), ty2.get_ty())
                return
            for (t1, t2) in zip(ty1.get_tys(), ty2.get_tys()):
                unify(t1, t2)
            return
        if ty1.is_fixed_len and not ty2.is_fixed_len:
            ty1.coerce_to_variable_len(ty2.get_ty())
            return
        if (not ty1.is_fixed_len) and ty2.is_fixed_len:
            ty2.coerce_to_variable_len(ty1.get_ty())
            return
        unify(ty1.get_ty(), ty2.get_ty())
        return

    if isinstance(ty1, TyDict) and isinstance(ty2, TyDict):
        unify(ty1.keyty, ty2.keyty)
        unify(ty1.valty, ty2.valty)
        return

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyTensor):
        utils.set_attr_if_None(ty1, ty2, 'kind')

        if ty1.dtype == ty2.dtype and ty1.ndim == ty2.ndim:
            if not inspect_shape:
                return
            unify_shape(ty1.shape, ty2.shape)
            return

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyNum):
        if ty1.ndim == 0:
            return

    if isinstance(ty1, TyNum) and isinstance(ty2, TyTensor):
        if ty2.ndim == 0:
            return

    if isinstance(ty1, TyDType) and isinstance(ty2, TyDType):
        utils.set_attr_if_None(ty1.dtype, ty2.dtype, 't')
        return

    if isinstance(ty1, TyUserDefinedClass) and \
            isinstance(ty2, TyUserDefinedClass):
        if ty1.name == ty2.name:
            return
        else:
            # TODO(momohatt): subtyping?
            raise UnifyError(ty1, ty2)

    raise UnifyError(ty1, ty2)
