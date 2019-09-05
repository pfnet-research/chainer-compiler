from enum import Enum, IntEnum

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

is_debug_global = False

def print_warning(msg):
    print("\x1b[33m[WARNING] " + msg + "\x1b[39m")

class TyObj():  # base type, meaning 'unknown'
    def __init__(self):
        self.is_optional = False
    # TODO(momohatt): fix __repr__
    def show(self):
        return "object"
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
    def __init__(self, ty_min, ty_max, value=None):
        super().__init__()
        assert ty_min <= ty_max
        self.ty_min = ty_min
        self.ty_max = ty_max
        self.value = value

    def show(self):
        return str(NumKind(self.ty_min))

    def __eq__(self, other):
        return isinstance(other, TyNum) and self.ty_min == other.ty_min

    def is_mutable(self):
        return False

    def possible_types(self):
        return list(range(self.ty_min, self.ty_max+ 1))

    def coerce_value(self):
        if self.value is None:
            return
        self.value = eval(self.show())(self.value)


def TyBool(value=None):
    return TyNum(0, 2, value=value)  # bool or int or float

def TyIntOnly():
    return TyNum(1, 1)  # int

def TyInt(value=None):
    return TyNum(1, 2, value=value)  # int or float

def TyFloat(value=None):
    return TyNum(2, 2, value=value)  # float


class TyString(TyObj):
    def __init__(self, value=None):
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
    def __init__(self, ty=None, seq_kind=None):
        super().__init__()
        self.seq_kind = seq_kind
        self.is_fixed_len = isinstance(ty, list) if ty is not None else None
        self.ty_ = ty

    def show(self):
        if self.is_fixed_len:
            if self.seq_kind == SequenceKind.LIST:
                return str(self.ty_)

            if self.seq_kind == SequenceKind.TUPLE:
                if len(self.ty_) == 0:
                    return "()"
                if len(self.ty_) == 1:
                    return "(" + str(self.ty_[0]) + ",)"
                return "(" + "".join([str(t) + ", " for t in self.ty_[:-1]]) \
                        + str(self.ty_[-1]) + ")"

            if len(self.ty_) == 0:
                return "{}"
            return "{" + "".join([str(t) + ", " for t in self.ty_[:-1]]) \
                    + str(self.ty_[-1]) + "}"

        if self.seq_kind == SequenceKind.LIST:
            return str(self.ty_) + " list"
        if self.seq_kind == SequenceKind.TUPLE:
            return str(self.ty_) + " tuple"

        return str(self.ty_) + " sequence"

    def __eq__(self, other):
        return isinstance(other, TySequence) and self.ty_ == other.ty_

    def is_mutable(self):
        return self.seq_kind == SequenceKind.LIST

    def unset(self):
        if self.is_fixed_len:
            for t in self.ty_:
                t.unset()
            return
        self.ty_.unset()

    def deref(self):
        if self.is_fixed_len is not None:
            if self.is_fixed_len:
                self.ty_ = [t.deref() for t in self.ty_]
            else:
                self.ty_ = self.ty_.deref()
        return self

    def get_ty(self):
        assert not self.is_fixed_len
        return self.ty_

    def get_tys(self):
        assert self.is_fixed_len
        return self.ty_

    def coerce_to_variable_len(self, ty=None):
        # does nothing if self is not fixed-length
        if self.is_fixed_len:
            if ty is None:
                ty = TyVar()
            for t in self.ty_:
                unify(ty, t)
            self.ty_ = ty
            self.is_fixed_len = False
        return

    def is_list(self):
        return self.seq_kind == SequenceKind.LIST

    def is_tuple(self):
        return self.seq_kind == SequenceKind.TUPLE


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
        if t is None:
            self.t = None
        else:
            self.t = np.dtype(t)
    def __str__(self):
        return "dtype({})".format(str(self.t))
    def __eq__(self, other):
        return self.t == other.t

    def is_float(self):
        return self.t in [np.dtype('float16'), np.dtype('float32'),
                np.dtype('float64'), np.dtype('float128')]


class TyTensor(TyObj):
    def __init__(self, dtype=None, kind=None, shape=None):  # we do not allow heterogeneous type ndarray
        # TODO(momohatt): shape
        super().__init__()
        assert isinstance(dtype, TyDType) or dtype is None
        self.dtype = dtype
        self.kind = kind
        self.shape = shape

    def show(self):
        if self.kind == TensorKind.ndarray:
            return "ndarray({}, shape={})".format(self.dtype, self.shape)
        if self.kind == TensorKind.chainer_variable:
            return "Variable({}, shape={})".format(self.dtype, self.shape)
        return "tensor({})".format(self.dtype)

    def __eq__(self, other):
        return isinstance(other, TyTensor) and self.dtype == other.dtype

    def is_mutable(self):
        return True

    def is_ndarray(self):
        return self.kind == TensorKind.ndarray

    def is_chainer_variable(self):
        return self.kind == TensorKind.chainer_variable


def TyNdarray(dtype=None, shape=None):
    return TyTensor(dtype=dtype, kind=TensorKind.ndarray, shape=shape)

def TyChainerVariable(dtype=None, shape=None):
    return TyTensor(dtype=dtype, kind=TensorKind.chainer_variable, shape=shape)


# ------------------------ TypeChecker internal types --------------------------

counter = 0

class TyVar(TyObj):
    def __init__(self):
        global counter
        super().__init__()
        self.i = counter
        counter += 1
        self.ty = None
        self.is_set = False

    def show(self):
        if self.ty:
            if is_debug_global:
                return "a{}({})".format(self.i, self.ty)
            return str(self.ty)
        return "a" + str(self.i)

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


def all_same_ty(tys):
    # TODO(momohatt): dtypeまで見る
    ty_tmp = TyVar()
    for t in tys:
        unify(ty_tmp, t)
    return True

def all_same(l):
    return all([e == l[0] for e in l])


def type_of_value(value) -> 'TyObj':
    # TODO: user defined class
    if value is None:
        return TyNone()
    if isinstance(value, bool):
        return TyBool()
    if isinstance(value, int):
        return TyInt()
    if isinstance(value, float):
        return TyFloat()
    if isinstance(value, str):
        return TyString()
    if isinstance(value, list):
        return TyList([type_of_value(v) for v in value])
    if isinstance(value, tuple):
        return TyTuple([type_of_value(v) for v in value])
    if isinstance(value, dict):
        return TyDict(type_of_value(list(value.keys())[0]),
                type_of_value(list(value.items())[0]))
    if isinstance(value, np.ndarray):
        return TyNdarray(dtype=TyDType(value.dtype), shape=value.shape)
    if isinstance(value, chainer.Variable):
        return TyChainerVariable(dtype=TyDType(value.dtype), shape=value.shape)
    if isinstance(value, L.Linear) or \
            isinstance(value, L.Convolution2D) or \
            isinstance(value, L.BatchNormalization):
        return TyArrow([TyChainerVariable(TyDType(np.float32))],
                TyChainerVariable(TyDType(np.float32)))
    if isinstance(value, L.NStepBiLSTM):
        # TODO(momohatt): allow other types
        return TyArrow([
            TyChainerVariable(TyDType(np.float32)),
            TyChainerVariable(TyDType(np.float32)),
            TyList(TyChainerVariable(TyDType(np.float32)))],
            TyTuple([
                TyChainerVariable(TyDType(np.float32)),
                TyChainerVariable(TyDType(np.float32)),
                TyList(TyChainerVariable(TyDType(np.float32)))]))
    if isinstance(value, np.dtype):
        return TyDType(value)
    if isinstance(value, type) and value in np.typeDict.values():
        # XXX: np.typeDict.values() is a list of all dtypes
        return TyDType(value)

    return TyUserDefinedClass(type(value).__name__, value)


def value_of_type(ty) -> object:
    ty = ty.deref()

    if isinstance(ty, TyNone):
        return None
    if isinstance(ty, TyNum):
        if ty.value is not None:
            return ty.value
        return pytype_of_type(ty)(1)  # XXX: to avoid division by zero
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
        ret = np.zeros(dtype=ty.dtype.t, shape=ty.shape)
        if ty.kind == TensorKind.ndarray:
            return ret
        if ty.kind == TensorKind.chainer_variable:
            return chainer.Variable(ret)
    if isinstance(ty, TyDType):
        return ty.t

    assert False, str(ty)


def pytype_of_type(ty) -> type:
    ty = ty.deref()

    if isinstance(ty, TyNum):
        return eval(ty.show())

    assert False


# ==============================================================================

class UnifyError(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "UnifyError: {} and {} are not unifiable".format(ty1, ty2)


def unify(ty1, ty2):
    def set_attr_if_None(obj1, obj2, attr_name):
        if hasattr(obj1, attr_name) and getattr(obj1, attr_name) is None:
            setattr(obj1, attr_name, getattr(obj2, attr_name))
            return
        if hasattr(obj2, attr_name) and getattr(obj2, attr_name) is None:
            setattr(obj2, attr_name, getattr(obj1, attr_name))
            return

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

    if type(ty1) is TyObj or type(ty2) is TyObj:
        return

    # if ty1 is not TyUnion, just do normal unification
    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return

    if isinstance(ty1, TyVar):
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
        possible_types = \
                [i for i in ty1.possible_types() if i in ty2.possible_types()]
        if possible_types == []:
            raise UnifyError(ty1, ty2)
        ty1.ty_min = ty2.ty_min = min(possible_types)
        ty1.ty_max = ty2.ty_max = max(possible_types)
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
        set_attr_if_None(ty1, ty2, 'ty_')

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
        # TODO(momohatt): coercion of dtype
        set_attr_if_None(ty1.dtype, ty2.dtype, 't')
        set_attr_if_None(ty1, ty2, 'kind')
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
