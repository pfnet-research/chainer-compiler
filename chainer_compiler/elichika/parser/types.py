from enum import Enum, IntEnum

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

is_debug_global = False

class Singleton():
    def __init__(self, x):
        self.inner = x
    def __eq__(self, other):
        return self.inner == other.inner
    def __str__(self):
        return str(self.inner)
    def __repr__(self):
        return self.__str__()


class TyObj():  # base type
    def __init__(self):
        self.is_optional = False
    # TODO(momohatt): fix __repr__
    def show(self):
        pass
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


numcounter = 0  # id for debug printing

class TyNum(TyObj):
    def __init__(self, ty_min, ty_max):
        global numcounter
        super().__init__()
        assert ty_min <= ty_max
        self.ty_min = Singleton(ty_min)
        self.ty_max = Singleton(ty_max)
        self.id = numcounter
        numcounter += 1

    def show(self):
        if is_debug_global:
            return "n{}({})".format(self.id, str(NumKind(self.ty_min.inner)))
        return str(NumKind(self.ty_min.inner))

    def __eq__(self, other):
        return isinstance(other, TyNum) and self.ty_min == other.ty_min

    def is_mutable(self):
        return False

    def possible_types(self):
        return list(range(self.ty_min.inner, self.ty_max.inner + 1))


def TyBool():
    return TyNum(0, 2)  # bool or int or float

def TyIntOnly():
    return TyNum(1, 1)  # int

def TyInt():
    return TyNum(1, 2)  # int or float

def TyFloat():
    return TyNum(2, 2)  # float


class TyString(TyObj):
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
    def __init__(self, ty, seq_kind=None):
        super().__init__()
        self.seq_kind = seq_kind
        self.is_fixed_len = isinstance(ty, list)
        self.ty_ = ty

    def show(self):
        if self.is_fixed_len:
            if self.seq_kind == SequenceKind.LIST:
                return str(self.ty_)

            if self.seq_kind == SequenceKind.TUPLE:
                if len(self.ty_) == 0:
                    return "()"
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
        return self.name


# --------------------- numpy ndarray / chainer variable -----------------------

class TensorKind(Enum):
    ndarray = 0
    chainer_variable = 1

class TyTensor(TyObj):
    def __init__(self, dtype=None, kind=None):  # we do not allow heterogeneous type ndarray
        super().__init__()
        self.dtype = Singleton(dtype)
        self.kind = kind

    def show(self):
        if self.kind == TensorKind.ndarray:
            return "ndarray(dtype={})".format(self.dtype)
        if self.kind == TensorKind.chainer_variable:
            return "Variable(dtype={})".format(self.dtype)
        return "tensor(dtype={})".format(self.dtype)

    def __eq__(self, other):
        return isinstance(other, TyTensor) and self.dtype == other.dtype

    def is_mutable(self):
        return True


def TyNdarray(dtype):
    return TyTensor(dtype, TensorKind.ndarray)

def TyChainerVariable(dtype):
    return TyTensor(dtype, TensorKind.chainer_variable)

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
    ty_tmp = TyVar()
    for t in tys:
        unify(ty_tmp, t)
    return True


def type_of_value(value) -> 'TyObj':
    # TODO: user defined class
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
        return TyDict(type_of_value(value.keys()[0]), type_of_value(value.items()[0]))
    if isinstance(value, np.ndarray):
        return TyNdarray(value.dtype)
    if isinstance(value, chainer.Variable):
        return TyChainerVariable(value.dtype)
    # TODO(momohatt): sometimes Linear's return type is tuple
    if isinstance(value, L.Linear):
        return TyArrow([TyChainerVariable(np.dtype('float32'))],
                TyChainerVariable(np.dtype('float32')))

    return TyUserDefinedClass(type(value).__name__, value)


def value_of_type(ty) -> object:
    ty = ty.deref()

    if isinstance(ty, TyNone):
        return None
    if isinstance(ty, TyNum):
        return pytype_of_type(ty)()
    if isinstance(ty, TyString):
        return ""
    if isinstance(ty, TySequence) and ty.is_fixed_len:
        ret = [value_of_type(t) for t in ty.get_tys()]
        if ty.is_list():
            return ret
        return tuple(ret)
    if isinstance(ty, TyDict):
        return { value_of_type(ty.keyty) : value_of_type(ty.valty) }
    if isinstance(ty, TyTensor):
        ret = np.array(0, dtype=ty.dtype.inner)
        if ty.kind == TensorKind.ndarray:
            return ret
        if ty.kind == TensorKind.chainer_variable:
            return chainer.Variable(ret)

    assert False


def pytype_of_type(ty) -> type:
    ty = ty.deref()

    if isinstance(ty, TyNum):
        return eval(str(ty))

    assert False


# ==============================================================================

class UnifyError(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "UnifyError: {} and {} are not unifiable".format(ty1, ty2)


def unify(ty1, ty2):
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
                print("\x1b[33m[LOG] unify error with " + str(ty1_) \
                        + " and " + str(ty2) + ". continuing...\x1b[39m")
                continue

        raise UnifyError(ty1, ty2)

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

    if isinstance(ty1, TyNum) and isinstance(ty2, TyNum):
        possible_types = \
                [i for i in ty1.possible_types() if i in ty2.possible_types()]
        if possible_types == []:
            raise UnifyError(ty1, ty2)
        ty1.ty_min.inner = ty2.ty_min.inner = min(possible_types)
        ty1.ty_max.inner = ty2.ty_max.inner = max(possible_types)
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
                raise UnifyError(ty1, ty2)
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
        if ty1.dtype.inner is None:
            ty1.dtype.inner = ty2.dtype.inner
        elif ty2.dtype.inner is None:
            ty2.dtype.inner = ty1.dtype.inner

        if ty1.kind is None:
            ty1.kind = ty2.kind
        elif ty2.kind is None:
            ty2.kind = ty1.kind
        return

    if isinstance(ty1, TyUserDefinedClass) and \
            isinstance(ty2, TyUserDefinedClass):
        if ty1.name == ty2.name:
            return
        else:
            # TODO(momohatt): subtyping?
            raise UnifyError(ty1, ty2)

    raise UnifyError(ty1, ty2)
