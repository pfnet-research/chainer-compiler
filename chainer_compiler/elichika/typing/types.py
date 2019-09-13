from   copy import deepcopy
from   enum import Enum, IntEnum

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from   chainer_compiler.elichika.typing import utils
from   chainer_compiler.elichika.typing.shape_elem import ShapeElem, wrap_shape, unwrap_shape

def print_warning(msg):
    print("\x1b[33m[WARNING] " + msg + "\x1b[39m")

class TyObj():  # base type, meaning 'unknown'
    def __init__(self):
        self.is_optional = False
    # TODO(momohatt): fix __repr__
    def __str__(self):
        if self.is_optional:
            return "optional({})".format(self.show())
        return self.show()
    def __repr__(self):
        return self.__str__()

    def is_mutable(self):
        pass
    def unset(self):
        return
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

    def unset(self):
        for t in self.argty:
            t.unset()
        self.retty.unset()

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

    def unset(self):
        if self.is_fixed_len:
            for t in self._ty:
                t.unset()
            return
        self._ty.unset()

    def deref(self):
        if self.is_fixed_len is not None:
            if self.is_fixed_len:
                self._ty = [t.deref() for t in self._ty]
            else:
                self._ty = self._ty.deref()
        return self

    def get(self):
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

    def unset(self):
        self.keyty.unset()
        self.valty.unset()

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

    def __eq__(self, other):
        # TODO: shape?
        return isinstance(other, TyTensor) and self.dtype == other.dtype

    def is_mutable(self):
        return True

    def is_ndarray(self):
        return self.kind == TensorKind.ndarray

    def is_chainer_variable(self):
        return self.kind == TensorKind.chainer_variable


def TyNdarray(dtype, ndim=None, shape=None):
    # ndim and shape cannot be None at the same time
    if ndim is None:
        ndim = len(shape)
    return TyTensor(dtype, TensorKind.ndarray, ndim=ndim, shape=shape)

def TyChainerVariable(dtype, ndim=None, shape=None):
    if ndim is None:
        ndim = len(shape)
    return TyTensor(dtype, TensorKind.chainer_variable, ndim=ndim, shape=shape)


# ------------------------ TypeChecker internal types --------------------------

counter = 0

class TyVar(TyObj):
    def __init__(self, lineno=None):
        global counter
        super().__init__()
        self.i = counter
        counter += 1
        self.ty = None
        self.is_set = False
        self.lineno = lineno

    def show(self):
        if self.ty:
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

    def unset(self):
        self.is_set = False
        self.ty = None

    def deref(self):
        if self.is_set:
            return self.ty.deref()
        return self


# TODO(momohatt): Deprecate TyUnion (and 'unset')
class TyUnion(TyObj):
    def __init__(self, *tys):
        assert len(tys) >= 2
        super().__init__()
        self.tys = list(tys)  # tys : tuple of TyObj
        self.is_set = False

    def show(self):
        if self.is_set:
            return str(self.tys)
        return str(self.tys[0]) + "".join([" \/ " + str(t) for t in self.tys[1:]])

    def set(self, ty):
        assert not self.is_set
        self.is_set = True
        self.tys = ty

    def deref(self):
        if self.is_set:
            self.tys = self.tys.deref()
            return self.tys

        self.tys = [t.deref() for t in self.tys]
        return self


# ------------------------------------------------------------------------------

def all_same_ty(tys):
    _tytmp = TyVar()
    for t in tys:
        unify(_tytmp, t)
    return True


def all_same(l):
    return all([e == l[0] for e in l])


def type_of_value(value) -> 'TyObj':
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
        return TyDict(type_of_value(list(value.keys())[0]),
                type_of_value(list(value.items())[0]))
    if isinstance(value, np.ndarray):
        return TyNdarray(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, chainer.Variable):
        return TyChainerVariable(value.dtype, shape=wrap_shape(value.shape))
    if isinstance(value, np.dtype):
        return TyDType(value)
    if isinstance(value, type) and value in np.typeDict.values():
        # XXX: np.typeDict.values() is a list of all dtypes
        return TyDType(value)
    if isinstance(value, ShapeElem):
        if isinstance(value.value, int):
            return TyInt(value.get_value())
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


def value_of_type(ty) -> object:
    # creates dummy value
    # TODO(momohatt): rename this function

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
            ret = [value_of_type(t) for t in ty.get_tys()]
        else:
            ret = [value_of_type(ty.get_ty())]
        if ty.is_list():
            return ret
        return tuple(ret)
    if isinstance(ty, TyDict):
        return { value_of_type(ty.keyty) : value_of_type(ty.valty) }
    if isinstance(ty, TyTensor):
        ret = np.zeros(dtype=ty.dtype, shape=unwrap_shape(ty.shape))
        if ty.is_ndarray():
            return ret
        if ty.is_chainer_variable():
            return chainer.Variable(ret)
    if isinstance(ty, TyDType):
        return ty.t

    assert False, "value_of_type: type not understood: " + str(ty)


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
    if isinstance(ty, TyNone) or isinstance(ty, TyNum) or \
            isinstance(ty, TyString):
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


# ==============================================================================

class UnifyError(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "UnifyError: {} and {} are not unifiable".format(ty1, ty2)


def set_attr_if_None(obj1, obj2, attr_name):
    if hasattr(obj1, attr_name) and getattr(obj1, attr_name) is None:
        setattr(obj1, attr_name, getattr(obj2, attr_name))
        return
    if hasattr(obj2, attr_name) and getattr(obj2, attr_name) is None:
        setattr(obj2, attr_name, getattr(obj1, attr_name))
        return


def unify(ty1, ty2, inspect_shape=True):
    # inspect_shape: shapeの同一性まで見るかどうか
    ty1 = ty1.deref()
    ty2 = ty2.deref()

    # XXX: TyUnion is only allowed in ty1
    # if ty1 is TyUnion, try unification one by one.
    if isinstance(ty1, TyUnion):
        for ty1_ in ty1.tys:
            try:
                unify(ty1_, ty2)
                ty1.set(ty1_)
                return
            except UnifyError:
                ty2.unset()
                print_warning("unify error with {} and {}. continuing...".format(
                    ty1_, ty2))
                continue

        raise UnifyError(ty1, ty2)

    # if ty1 is not TyUnion, just do normal unification
    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return

    if isinstance(ty1, TyVar):
        if isinstance(ty2, TyVar) and ty1 is ty2:
            return
        # TODO(momohatt): occur check
        ty1.set(ty2)
        return

    if isinstance(ty2, TyVar):
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
        set_attr_if_None(ty1, ty2, 'is_fixed_len')
        set_attr_if_None(ty1, ty2, '_ty')

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
            for ty in ty1.get_tys():
                unify(ty, ty2.get_ty())
            ty1.coerce_to_variable_len(ty2.get_ty())
            return
        if (not ty1.is_fixed_len) and ty2.is_fixed_len:
            unify(ty2, ty1)
            return
        unify(ty2.get_ty(), ty1.get_ty())
        return

    if isinstance(ty1, TyDict) and isinstance(ty2, TyDict):
        unify(ty1.keyty, ty2.keyty)
        unify(ty1.valty, ty2.valty)
        return

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyTensor):
        set_attr_if_None(ty1, ty2, 'dtype')
        set_attr_if_None(ty1, ty2, 'kind')

        if ty1.dtype == ty2.dtype and ty1.ndim == ty2.ndim:
            if not inspect_shape:
                return
            try:
                for s1, s2 in zip(ty1.shape, ty2.shape):
                    unify_shapeElem(s1, s2)
                return
            except Exception:
                raise UnifyError(ty1, ty2)

    if isinstance(ty1, TyTensor) and isinstance(ty2, TyNum):
        if ty1.ndim == 0:
            return

    if isinstance(ty1, TyNum) and isinstance(ty2, TyTensor):
        if ty2.ndim == 0:
            return

    if isinstance(ty1, TyDType) and isinstance(ty2, TyDType):
        set_attr_if_None(ty1.dtype, ty2.dtype, 't')
        return

    if isinstance(ty1, TyUserDefinedClass) and \
            isinstance(ty2, TyUserDefinedClass):
        if ty1.name == ty2.name:
            return
        else:
            # TODO(momohatt): subtyping?
            raise UnifyError(ty1, ty2)

    raise UnifyError(ty1, ty2)


def unify_shapeElem(e1, e2):
    if e1.value and e2.value:
        if e1.value != e2.value:
            raise Exception

    set_attr_if_None(e1, e2, 'value')
