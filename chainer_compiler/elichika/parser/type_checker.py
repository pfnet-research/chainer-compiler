import ast
import inspect
import gast
import os
import traceback
import types
from copy import deepcopy
from typing import List
from pprint import pprint

from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser.types import *

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

# ============================ Display utils ===================================

def intercalate(strings, sep):
    return "".join([s + sep for s in strings[:-1]]) + strings[-1]


def expr_to_str(node):
    if isinstance(node, gast.BoolOp):
        return intercalate([expr_to_str(e) for e in node.values],
                " " + boolop_to_str(node.op) + " ")
    if isinstance(node, gast.BinOp):
        return "{} {} {}".format(expr_to_str(node.left),
                operator_to_str(node.op), expr_to_str(node.right))
    if isinstance(node, gast.Call):
        return "{}({})".format(expr_to_str(node.func),
                intercalate([expr_to_str(arg) for arg in node.args], ", "))
    if isinstance(node, gast.Num):
        return str(node.n)
    if isinstance(node, gast.Str):
        return str(node.s)
    if isinstance(node, gast.Attribute):
        return "{}.{}".format(expr_to_str(node.value), node.attr)
    if isinstance(node, gast.Name):
        return node.id
    return ""


def boolop_to_str(node):
    if isinstance(node, gast.And):
        return "and"
    if isinstance(node, gast.Or):
        return "or"


def operator_to_str(node):
    if isinstance(node, gast.Add):
        return "+"
    if isinstance(node, gast.Sub):
        return "-"
    if isinstance(node, gast.Mult):
        return "*"
    if isinstance(node, gast.Div):
        return "/"
    if isinstance(node, gast.FloorDiv):
        return "//"


def is_expr(node):
    return isinstance(node, gast.BoolOp) \
            or isinstance(node, gast.BinOp) \
            or isinstance(node, gast.UnaryOp) \
            or isinstance(node, gast.Lambda) \
            or isinstance(node, gast.IfExp) \
            or isinstance(node, gast.Dict) \
            or isinstance(node, gast.Set) \
            or isinstance(node, gast.ListComp) \
            or isinstance(node, gast.SetComp) \
            or isinstance(node, gast.DictComp) \
            or isinstance(node, gast.GeneratorExp) \
            or isinstance(node, gast.Await) \
            or isinstance(node, gast.Yield) \
            or isinstance(node, gast.YieldFrom) \
            or isinstance(node, gast.Compare) \
            or isinstance(node, gast.Call) \
            or isinstance(node, gast.Repr) \
            or isinstance(node, gast.Num) \
            or isinstance(node, gast.Str) \
            or isinstance(node, gast.FormattedValue) \
            or isinstance(node, gast.JoinedStr) \
            or isinstance(node, gast.Bytes) \
            or isinstance(node, gast.NameConstant) \
            or isinstance(node, gast.Ellipsis) \
            or isinstance(node, gast.Attribute) \
            or isinstance(node, gast.Subscript) \
            or isinstance(node, gast.Starred) \
            or isinstance(node, gast.Name) \
            or isinstance(node, gast.List) \
            or isinstance(node, gast.Tuple)


def node_description(node):
    type_name = type(node).__name__
    lineno = " (line {})".format(node.lineno) if hasattr(node, 'lineno') else ""
    if isinstance(node, gast.FunctionDef):
        return "{} {}{}".format(type_name, node.name, lineno)
    if is_expr(node):
        return "{} {}{}".format(type_name, expr_to_str(node), lineno)
    return type_name

# ==============================================================================

def debug(sth):
    frame = inspect.currentframe().f_back
    print("[{} {}] {}".format(frame.f_code.co_name, frame.f_lineno, sth))


def add_dict(dest, src):
    for k, v in src.items():
        dest[k] = v


def defined_with___call__(func):
    return not isinstance(func, (types.FunctionType, types.MethodType,
        types.BuiltinFunctionType))


def callable_(x):
    if isinstance(x, L.Linear) or \
            isinstance(x, L.Convolution2D) or \
            isinstance(x, L.BatchNormalization):
        return False
    return callable(x)


def copy_ty(ty):
    if isinstance(ty, TyUserDefinedClass):
        # XXX: do not copy instance
        return TyUserDefinedClass(ty.name, ty.instance)
    return deepcopy(ty)


def copy_tyenv(tyenv):
    new_tyenv = {}
    for name, ty in tyenv.items():
        new_tyenv[name] = copy_ty(ty)
    return new_tyenv


# ==============================================================================

builtins_ty = {
        float : TyArrow([TyBool()], TyFloat()),
        # int -> int list \/ int -> int -> int list \/
        # int -> int -> int -> int list
        range : TyUnion(
            TyArrow([TyIntOnly()], TyList(TyIntOnly())),
            TyArrow([TyIntOnly(), TyIntOnly()], TyList(TyIntOnly())),
            TyArrow([TyIntOnly(), TyIntOnly(), TyIntOnly()], TyList(TyIntOnly())),
            ),
        # let x = ... in TyArrow([x], x)
        abs : TyUnion(
            (lambda x: TyArrow([x], x))(TyNum(0, 2)),
            (lambda x: TyArrow([x], x))(TyTensor())
            ),
        len : TyUnion(
            TyArrow([TySequence()], TyInt()),
            TyArrow([TyDict(TyVar(), TyVar())], TyInt()),
            TyArrow([TyString()], TyInt()),
            ),
        }

builtins_name = [f.__name__ for f in builtins_ty.keys()]


def make_infer(func, fallback_shapes, fallback_dtypes):
    def infer(ty_args, dummy_args_nontensor, kwargs):
        ty_args_tensor = [t for t in ty_args if isinstance(t, TyTensor)]

        shapes = [s if t.shape is None else t.shape
                for t, s in zip(ty_args_tensor, fallback_shapes)]
        dtypes = [s if t.dtype.t is None else t.dtype.t
                for t, dt in zip(ty_args_tensor, fallback_dtypes)]
        is_dummy_shape = any([t.shape is None for t in ty_args_tensor])
        is_dummy_dtype = any([t.dtype.t is None for t in ty_args_tensor])
        # XXX: tensor arguments always come before non-tensor arguments
        dummy_args = [np.zeros(s, t) for s, t in zip(shapes, dtypes)] + \
                dummy_args_nontensor
        dummy_result = func(*dummy_args, **kwargs)
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
    def infer(ty_args, dummy_args_nontensor, kwargs):
        ksize = dummy_args_nontensor[0]
        # TODO(momohatt): handle cases where stride is not specified as kwarg
        # but arg
        stride = kwargs['stride'] if 'stride' in kwargs.items() else ksize
        minimum_size = max(ksize, stride)
        fallback_shapes = ((1, 1, minimum_size, minimum_size),)
        fallback_dtypes = (np.float32,)

        return make_infer(func, fallback_shapes, fallback_dtypes) \
                (ty_args, dummy_args_nontensor, kwargs)

    return infer


def ty_ChainerSoftmaxCrossEntropy(ty_args, dummy_args_nontensor, kwargs):
    shape_x, shape_t = ty_args[0].shape, ty_args[1].shape
    fallback_dtypes = (np.float32, np.int64)

    # x.shape[0] == t.shape[0]
    if shape_x is None and shape_t is None:
        fallback_shapes = ((1, 1), (1,))
    elif shape_x is None:
        fallback_shapes = ((shape_t[0], 1), shape_t)
    elif shape_t is None:
        fallback_shapes = (shape_x, (shape_x[0],))

    return make_infer(
            F.softmax_cross_entropy, fallback_shapes, fallback_dtypes) \
                    (ty_args, dummy_args_nontensor, kwargs)


# math functions that doesn't change shapes or dtypes
def ty_ChainerIdentical(ty_args, dummy_args_nontensor, kwargs):
    if isinstance(ty_args[0], TyTensor):
        assert ty_args[0].dtype.is_float()
        return ty_args[0]
    assert False


def ty_ChainerConcat(ty_args, dummy_args_nontensor, kwargs):
    # TODO(momohatt): shape
    assert isinstance(ty_args[0], TySequence)
    if ty_args[0].is_fixed_len:
        dtypes = [tytensor.dtype for tytensor in ty_args[0].get_tys()]
        assert all_same(dtypes)
        return TyTensor(dtype=dtypes[0])

    dtype = ty_args[0].get_ty().dtype
    return TyTensor(dtype=dtype)


def ty_ChainerExpandDims(ty_args, dummy_args_nontensor, kwargs):
    # TODO(momohatt): axis can come as dummy_args_nontensor
    axis = kwargs['axis'] if dummy_args_nontensor == [] else \
            dummy_args_nontensor[0]
    fallback_shapes = ((1,) * axis,)
    fallback_dtypes = (np.float32,)

    return make_infer(F.expand_dims, fallback_shapes, fallback_dtypes) \
            (ty_args, dummy_args_nontensor, kwargs)


def ty_ChainerBroadcastTo(ty_args, dummy_args_nontensor, kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSum(ty_args, dummy_args_nontensor, kwargs):
    axis = kwargs['axis'] if dummy_args_nontensor == [] else \
            dummy_args_nontensor[0]
    fallback_shapes = ((1,) * (axis + 1),)
    fallback_dtypes = (np.float32,)

    return make_infer(F.sum, fallback_shapes, fallback_dtypes) \
            (ty_args, dummy_args_nontensor, kwargs)


ext_func_ty = {
        np.array : evaluate_function_types(
            np.array, 0),
        np.ones : evaluate_function_types(
            np.ones, 0),
        np.zeros : evaluate_function_types(
            np.zeros, 0),
        chainer.Variable : evaluate_function_types(
            chainer.Variable, 1),
        F.average_pooling_2d :
            ty_ChainerPooling2d(F.average_pooling_2d),
        F.broadcast_to :
            ty_ChainerBroadcastTo,
        F.concat :
            ty_ChainerConcat,
        F.dropout :
            ty_ChainerIdentical,
        F.expand_dims :
            ty_ChainerExpandDims,
        F.local_response_normalization : evaluate_function_types(
            F.local_response_normalization, 1),
        F.max_pooling_2d :
            ty_ChainerPooling2d(F.max_pooling_2d),
        F.pad_sequence : evaluate_function_types(
            F.pad_sequence, 0),
        F.relu : evaluate_function_types(
            F.relu, 1),
        F.reshape :
            # TODO(momohatt): infer shape
            lambda ty_args, dummy_args_nontensor, kwargs: ty_args[0],
        F.softmax :
            ty_ChainerIdentical,
        F.softmax_cross_entropy :
            ty_ChainerSoftmaxCrossEntropy,
        F.sum :
            ty_ChainerSum,
        F.tanh :
            ty_ChainerIdentical,
        }


list_attr_ty = {
        'append'  : lambda x: TyArrow([x.get_ty()], TyNone()),
        'reverse' : lambda x: TyArrow([], TyNone()),
        }


def evaluate_binop_ty(op, tyl, tyr):
    semantics = {
            gast.Add : (lambda x, y: x + y),
            gast.Sub : (lambda x, y: x - y),
            gast.Mult : (lambda x, y: x * y),
            gast.Div : (lambda x, y: x / y),
            gast.FloorDiv : (lambda x, y: x // y),
            }
    func = semantics[type(op)]
    vall, valr = value_of_type(tyl), value_of_type(tyr)
    ty_ret = type_of_value(func(vall, valr))
    if isinstance(ty_ret, TySequence) and \
            not (tyl.is_fixed_len and tyr.is_fixed_len):
        ty_ret.coerce_to_variable_len()

    if isinstance(ty_ret, TyTensor) and \
            isinstance(tyl, TyTensor) and tyl.shape is None or \
            isinstance(tyr, TyTensor) and tyr.shape is None:
        ty_ret.shape = None
    return ty_ret


# ==============================================================================

class TypeChecker():
    class ArgumentRequired(Exception):
        def __init__(self, func=None, ty_obj=None):
            self.func = func  # callables
            self.ty_obj = ty_obj  # method call against

    def __init__(self, tyenv=None, attribute_tyenv=None, is_debug=False, module=None):
        # string -> TyObj (internal type env)
        self.tyenv = {} if tyenv is None else copy_tyenv(tyenv)
        # type environments
        self.nodetype = {}  # Node -> TyObj (for elichika to use)
        self.is_debug = is_debug
        self.module = module
        self.subroutine_node = {}  # Node (Call) -> Node (FunctionDef)

        # types of object attributes which are overwritten in forward()
        # (object, str) -> TyObj)
        self.attribute_tyenv = {} if attribute_tyenv is None \
                else copy_tyenv(attribute_tyenv)

        # True iff the parent node is Call
        self.is_function = False


    def dump_tyenv(self):
        print("=== tyenv ===")
        if not self.is_debug:
            return
        for name, ty in self.tyenv.items():
            print("{} : \x1b[35m{}\x1b[39m".format(name, ty))
        for (obj, name), ty in self.attribute_tyenv.items():
            # XXX: remove attributes inherited from libraries
            if name[0] == '_': continue
            print("self.{} : \x1b[35m{}\x1b[39m".format(name, ty))
        print()


    def dump_nodetype(self):
        if not self.is_debug:
            return
        for node, ty in self.nodetype.items():
            print("{} : \x1b[36m{}\x1b[39m".format(node_description(node), ty))
        print()


    def get_kwarg(self, keywords):
        ret = {}
        for k in keywords:
            ret[k.arg] = self.evaluate(k.value)
        return ret


    def evaluate(self, node):
        if isinstance(node, gast.Attribute):
            v_value = self.evaluate(node.value)
            if v_value is None:
                return None
            attr = getattr(v_value, node.attr)
            return attr

        if isinstance(node, gast.Num):
            return node.n

        if isinstance(node, gast.Str):
            return node.s

        if isinstance(node, gast.Name) and hasattr(self.module, node.id):
            return getattr(self.module, node.id)



    def infer(self, node):
        """
        Adds local type information to self.tyenv while traversing the AST
        while inlining functions and rewriting the argument 'node'
        returns: type
        """
        self.infer_mod(node)

        if self.is_debug:
            print('=== Type Environment ===')
            self.dump_nodetype()

        return self.nodetype


    def infer_function_vargs(self, node, args: List[object]):
        # args: argument value
        ty_args = [type_of_value(arg) for arg in args]
        return self.infer_function(node, ty_args)


    def infer_function(self, node, ty_args: List['TyObj']):
        # TODO(momohatt): varargs
        assert isinstance(node, gast.FunctionDef)
        if node.args.vararg == []:
            assert len(ty_args) == len(node.args.args), \
                    "Wrong number of arguments: expected {}, got {}".format(
                            len(node.args.args), len(ty_args))

        for arg_node, ty in zip(node.args.args, ty_args):
            self.tyenv[arg_node.id] = ty
        for ty in ty_args:
            if isinstance(ty, TyUserDefinedClass):
                for attr, val in ty.instance.__dict__.items():
                    self.attribute_tyenv[(ty.instance, attr)] = \
                            type_of_value(val)

        self.infer_stmt(node)

        if self.is_debug:
            print('=== Type Environment ===')
            self.dump_nodetype()

        return self.nodetype


    def infer_block(self, stmts):  # use in if (without else), for, while
        tc = TypeChecker(
                tyenv=self.tyenv, attribute_tyenv=self.attribute_tyenv,
                is_debug=self.is_debug, module=self.module)
        for stmt in stmts:
            tc.infer_stmt(stmt)

        # 1. unify the intersection of 2 tyenvs and update local tyenv
        for name, ty in tc.tyenv.items():
            if name in self.tyenv.keys():
                unify(ty, self.tyenv[name])
            self.tyenv[name] = ty

        for (obj, name), ty in tc.attribute_tyenv.items():
            if (obj, name) in self.attribute_tyenv.keys():
                unify(ty, self.attribute_tyenv[(obj, name)])
            self.attribute_tyenv[(obj, name)] = ty

        # 2. merge nodetype from 2 TypeCheckers
        add_dict(self.nodetype, tc.nodetype)
        add_dict(self.subroutine_node, tc.subroutine_node)


    # ================================ mod =====================================
    def infer_mod(self, node):
        if isinstance(node, gast.Module):
            self.infer_stmt(node.body[0])
            return


    # ================================ stmt ====================================
    def infer_stmt(self, node) -> 'TyObj':
        if self.is_debug:
            debug(gast.dump(node))

        if isinstance(node, gast.FunctionDef):
            # FunctionDef(identifier name, arguments args, stmt* body,
            # expr* decorator_list, expr? returns)

            ty_args = [self.tyenv[arg.id] for arg in node.args.args]
            ty = None

            for stmt in node.body:
                ty = self.infer_stmt(stmt)

            assert ty is not None
            self.nodetype[node] = TyArrow(ty_args, ty)
            return self.nodetype[node]


        if isinstance(node, gast.Return):
            # Return(expr? value)
            self.nodetype[node] = self.infer_expr(node.value)
            return self.nodetype[node]


        if isinstance(node, gast.Assign):
            # Assign(expr* targets, expr value)
            assert len(node.targets) == 1  # cannot think of cases where >= 2
            target = node.targets[0]
            ty_val = self.infer_expr(node.value)

            if isinstance(target, gast.Name):
                if (isinstance(node.value, gast.Name) or \
                        isinstance(node.value, gast.Attribute)) and \
                        ty_val.is_mutable():
                    # XXX: alias
                    self.tyenv[target.id] = ty_val
                    self.nodetype[target] = ty_val
                else:
                    self.tyenv[target.id] = copy_ty(ty_val)
                    self.nodetype[target] = copy_ty(ty_val)

            elif isinstance(target, gast.Attribute):
                self.infer_expr(target)
                ty_obj = self.nodetype[target.value]
                assert isinstance(ty_obj, TyUserDefinedClass)
                self.attribute_tyenv[(ty_obj.instance, target.attr)] = ty_val

            elif type(target) in [gast.Tuple, gast.List]:
                ty_target = self.infer_expr(target)
                unify(ty_target, ty_val)
                for (var, ty) in zip(target.elts, ty_val.get_tys()):
                    self.tyenv[var.id] = ty
                    self.nodetype[var] = ty
            else:
                assert False

            self.nodetype[node] = TyNone()
            return self.nodetype[node]


        if isinstance(node, gast.AugAssign):
            # AugAssign(expr target, operator op, expr value)
            binop = gast.BinOp(node.target, node.op, node.value)
            ty_val = self.infer_expr(binop)
            ty_target = self.infer_expr(node.target)
            del self.nodetype[binop]
            if ty_target.is_mutable():
                unify(ty_target, ty_val)

            if isinstance(node.target, gast.Name):
                if ty_target.is_mutable():
                    self.tyenv[node.target.id] = ty_val
                else:
                    self.tyenv[node.target.id] = copy_ty(ty_val)

            if isinstance(node.target, gast.Attribute):
                ty_obj = self.nodetype[node.target.value]
                assert isinstance(ty_obj, TyUserDefinedClass)
                if ty_target.is_mutable():
                    self.attribute_tyenv[(ty_obj.instance, node.target.attr)] = \
                            ty_val
                else:
                    self.attribute_tyenv[(ty_obj.instance, node.target.attr)] = \
                            copy_ty(ty_val)

            self.nodetype[node.target] = ty_val
            self.nodetype[node] = TyNone()
            return self.nodetype[node]


        if isinstance(node, gast.For):
            # For(expr target, expr iter, stmt* body, stmt* orelse)
            assert type(node.target) in [gast.Name, gast.Tuple]

            ty_iteration = self.infer_expr(node.iter)
            ty_i = self.infer_expr(node.target)
            unify(ty_iteration, TyList(ty_i))

            # TODO(momohatt): scope of iteration variable is wrong
            self.infer_block(node.body)

            self.nodetype[node] = TyNone()
            return self.nodetype[node]


        if isinstance(node, gast.While):
            # While(expr test, stmt* body, stmt* orelse)
            # TODO
            return self.nodetype[node]


        if isinstance(node, gast.If):
            # If(expr test, stmt* body, stmt* orelse)
            ty_test = self.infer_expr(node.test)
            # TODO(momohatt): determine what type should ty_test be

            if node.orelse == []:
                self.infer_block(node.body)
            else:
                tc1 = TypeChecker(
                        tyenv=self.tyenv, attribute_tyenv=self.attribute_tyenv,
                        is_debug=self.is_debug, module=self.module)
                tc2 = TypeChecker(
                        tyenv=self.tyenv, attribute_tyenv=self.attribute_tyenv,
                        is_debug=self.is_debug, module=self.module)
                for stmt in node.body:
                    tc1.infer_stmt(stmt)
                for stmt in node.orelse:
                    tc2.infer_stmt(stmt)

                # 1. unify the intersection of 2 tyenvs and update local tyenv
                for name, ty in tc1.tyenv.items():
                    if name in tc2.tyenv.keys():
                        unify(ty, tc2.tyenv[name])
                        # XXX: objects existing in only one branch should not
                        # remain
                        self.tyenv[name] = ty

                for (obj, name), ty in tc1.attribute_tyenv.items():
                    if (obj, name) in tc2.attribute_tyenv.keys():
                        unify(ty, tc2.attribute_tyenv[(obj, name)])
                        self.attribute_tyenv[(obj, name)] = ty

                # 2. merge nodetype from 2 TypeCheckers
                add_dict(self.nodetype, tc1.nodetype)
                add_dict(self.nodetype, tc2.nodetype)
                add_dict(self.subroutine_node, tc1.subroutine_node)
                add_dict(self.subroutine_node, tc2.subroutine_node)


            self.nodetype[node] = TyNone()
            return self.nodetype[node]


        if isinstance(node, gast.Expr):
            # Expr(expr value)
            return TyNone()


        if isinstance(node, gast.Pass):
            self.nodetype[node] = TyNone()
            return self.nodetype[node]

        assert False, type(node).__name__


    # ================================= expr ===================================
    def infer_expr(self, node) -> 'TyObj':
        if self.is_debug:
            debug(gast.dump(node))

        if isinstance(node, gast.BoolOp):
            # BoolOp(boolop op, expr* values)
            ty_vals = [self.infer_expr(val) for val in node.values]
            for ty in ty_vals:
                unify(ty, TyBool())
            self.nodetype[node.op] = TyArrow([TyBool(), TyBool()], TyBool())
            self.nodetype[node] = TyBool()
            return self.nodetype[node]


        if isinstance(node, gast.BinOp):
            # BinOp(expr left, operator op, expr right)
            tyl = self.infer_expr(node.left).deref()
            tyr = self.infer_expr(node.right).deref()
            ty_ret = evaluate_binop_ty(node.op, tyl, tyr)
            self.nodetype[node.op] = TyArrow([tyl, tyr], ty_ret)
            self.nodetype[node] = ty_ret
            return self.nodetype[node]

        if isinstance(node, gast.UnaryOp):
            # UnaryOp(unaryop op, expr operand)
            if isinstance(node.op, gast.Invert):
                pass
            elif isinstance(node.op, gast.Not):
                pass
            elif isinstance(node.op, gast.UAdd):
                pass
            elif isinstance(node.op, gast.USub):
                # TODO(momohatt): UnaryOp(op=USub(), operand=Num(n=1)) should be
                # canonicalized to Num(n=-1)?

                ty_expr = self.infer_expr(node.operand)
                self.nodetype[node] = ty_expr  # TODO: fix this
            return self.nodetype[node]


        if isinstance(node, gast.Dict):
            # Dict(expr* keys, expr* values)
            if node.keys == []:
                self.nodetype[node] = TyDict(TyVar(), TyVar())
            else:
                ty_keys = [self.infer_expr(key) for key in node.keys]
                ty_vals = [self.infer_expr(val) for val in node.values]
                assert all_same_ty(ty_keys)
                assert all_same_ty(ty_vals)
                self.nodetype[node] = TyDict(ty_keys[0], ty_vals[0])
            return self.nodetype[node]


        if isinstance(node, gast.Compare):
            # Compare(expr left, cmpop* ops, expr* comparators)
            # TODO
            self.nodetype[node] = TyBool()
            return self.nodetype[node]


        if isinstance(node, gast.Call):
            return self.infer_Call(node)

        if isinstance(node, gast.Num):
            # Num(object n)
            if isinstance(node.n, int):
                self.nodetype[node] = TyInt()
            elif isinstance(node.n, float):
                self.nodetype[node] = TyFloat()
            return self.nodetype[node]


        if isinstance(node, gast.Str):
            # Str(string s)
            self.nodetype[node] = TyString()
            return self.nodetype[node]


        if isinstance(node, gast.NameConstant):
            # NameConstant(singleton value)
            # value is either True, False or None
            if isinstance(node.value, bool):
                self.nodetype[node] = TyBool()
            elif node.value is None:
                self.nodetype[node] = TyNone()
            return self.nodetype[node]


        if isinstance(node, gast.Attribute):
            return self.infer_Attribute(node)

        if isinstance(node, gast.Subscript):
            # Subscript(expr value, slice slice, expr_context ctx)
            ty_obj = self.infer_expr(node.value)

            if isinstance(ty_obj, TySequence):
                self.infer_slice(node.slice, TyInt())
                if ty_obj.is_fixed_len and \
                        isinstance(node.slice, gast.Index) and \
                        isinstance(node.slice.value, gast.Num):
                    # TODO(momohatt): handle cases where index is
                    # more complex but still a constant
                    self.nodetype[node] = ty_obj.get_tys()[node.slice.value.n]
                    return self.nodetype[node]

                ty_obj.coerce_to_variable_len()
                if isinstance(node.slice, gast.Index):
                    self.nodetype[node] = ty_obj.get_ty()
                elif isinstance(node.slice, gast.Slice):
                    self.nodetype[node] = ty_obj
                else:
                    assert False
                return self.nodetype[node]

            if isinstance(ty_obj, TyDict):
                self.infer_slice(node.slice, ty_obj.keyty)
                assert isinstance(node.slice, gast.Index)
                self.nodetype[node] = ty_obj.valty
                return self.nodetype[node]

            if isinstance(ty_obj, TyNdarray):
                self.infer_slice(node.slice, TyInt())
                if isinstance(node.slice, gast.Index):
                    self.nodetype[node] = ty_obj.ty
                elif isinstance(node.slice, gast.Slice):
                    self.nodetype[node] = ty_obj
                else:
                    assert False
                return self.nodetype[node]

            else:
                assert False


        if isinstance(node, gast.Name):
            # Name(identifier id, expr_context ctx, expr? annotation)
            if node.id in self.tyenv.keys():
                self.nodetype[node] = self.tyenv[node.id]
            elif node.id in builtins_name:
                self.nodetype[node] = copy_ty(builtins_ty[eval(node.id)])
            elif hasattr(self.module, node.id):
                x = getattr(self.module, node.id)
                if callable_(x):
                    raise self.ArgumentRequired(func=x)
                self.nodetype[node] = type_of_value(x)
            else:
                # case of Tuple assignment
                ty_var = TyVar()
                self.tyenv[node.id] = ty_var
                self.nodetype[node] = ty_var
            return self.nodetype[node]


        if isinstance(node, gast.List):
            # List(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyList(elts_ty)
            return self.nodetype[node]


        if isinstance(node, gast.Tuple):
            # Tuple(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyTuple(elts_ty)
            return self.nodetype[node]

        assert False, type(node).__name__


    def infer_Call(self, node):
        # Call(expr func, expr* args, keyword* keywords)
        ty_args = [self.infer_expr(arg) for arg in node.args]
        ty_ret = TyVar()

        try:
            self.is_function = True
            ty_fun = self.infer_expr(node.func)
            self.is_function = False

        except self.ArgumentRequired as e:
            self.is_function = False
            if e.func is None:
                # attribute against tensor etc.
                assert isinstance(node.func, gast.Attribute)
                ty_obj = self.nodetype[node.func.value]

                if isinstance(ty_obj, TyTensor) and ty_obj.is_ndarray():
                    if node.func.attr == 'astype':
                        val_args = [self.evaluate(arg) for arg in node.args]
                        ty_ret = TyNdarray(TyDType(val_args[0]))
                    self.nodetype[node] = ty_ret
                    self.nodetype[node.func] = TyArrow(ty_args, ty_ret)

                return self.nodetype[node]

            if e.func in ext_func_ty.keys():
                # case of calling external (eg. np/chainer) functions

                # Non-tensor arguments
                val_dummy_args_nontensor = []
                for t, arg in zip(ty_args, node.args):
                    if isinstance(t, TyTensor):
                        continue
                    v = self.evaluate(arg)
                    val_dummy_args_nontensor.append(v if v is not None else value_of_type(t))
                val_kwargs = self.get_kwarg(node.keywords)
                inference_logic = ext_func_ty[e.func]
                ty_ret = inference_logic(
                        ty_args, val_dummy_args_nontensor, val_kwargs)

                self.nodetype[node] = ty_ret
                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
                return self.nodetype[node]

            if isinstance(e.func, types.BuiltinFunctionType):
                # TODO
                assert False

            # user defined functions, need to inline
            if isinstance(e.func, types.FunctionType) or \
                    isinstance(e.func, types.MethodType):
                code = utils.clip_head(inspect.getsource(e.func))

                if isinstance(node.func, gast.Attribute):
                    ty_self = self.nodetype[node.func.value]
                    ty_args = [ty_self] + ty_args

            else:
                # defined with __call__
                if isinstance(e.func, chainer.Chain):
                    code = utils.clip_head(inspect.getsource(e.func.forward))
                else:
                    code = utils.clip_head(inspect.getsource(e.func.__call__))
                ty_self = self.nodetype[node.func]
                ty_args = [ty_self] + ty_args

            # FunctionDef of called subroutine
            func_node = gast.ast_to_gast(ast.parse(code)).body[0]
            self.subroutine_node[node] = func_node
            tc = TypeChecker(is_debug=self.is_debug, module=self.module)
            tc.infer_function(func_node, ty_args)

            # copy nodetype and subroutine_node from subroutine
            add_dict(self.nodetype, tc.nodetype)
            add_dict(self.subroutine_node, tc.subroutine_node)

            ty_fun = tc.nodetype[func_node]
            self.nodetype[node.func] = ty_fun


        unify(ty_fun, TyArrow(ty_args, ty_ret))
        self.nodetype[node] = ty_ret.deref()

        return self.nodetype[node]


    def infer_Attribute(self, node):
        # Attribute(expr value, identifier attr, expr_context ctx)

        if isinstance(node.value, gast.Name) and \
                hasattr(self.module, node.value.id):
            # function of imported libraries (eg. np, chainer, F, L)
            module = getattr(self.module, node.value.id)
            attr = getattr(module, node.attr)
            if callable_(attr) and self.is_function:
                raise self.ArgumentRequired(func=attr)
            self.nodetype[node] = type_of_value(attr)
            return self.nodetype[node]

        ty_obj = self.infer_expr(node.value)

        if isinstance(ty_obj, TySequence) and ty_obj.is_list():
            ty_obj.coerce_to_variable_len()
            self.nodetype[node] = list_attr_ty[node.attr](ty_obj)
            return self.nodetype[node]

        if isinstance(ty_obj, TyTensor):
            if node.attr == 'shape':
                if ty_obj.shape is None:
                    self.nodetype[node] = TyTuple(TyInt())
                else:
                    self.nodetype[node] = type_of_value(ty_obj.shape)
                return self.nodetype[node]
            if ty_obj.is_ndarray() and node.attr == 'astype':
                raise self.ArgumentRequired()
            assert False

        if isinstance(ty_obj, TyUserDefinedClass):
            # x: value of existing instance
            x = getattr(ty_obj.instance, node.attr)

            if (ty_obj.instance, node.attr) in self.attribute_tyenv.keys():
                self.nodetype[node] = \
                        self.attribute_tyenv[(ty_obj.instance, node.attr)]
            else:
                self.nodetype[node] = type_of_value(x)

            if callable_(x) and self.is_function:
                if x in builtins_ty.keys():
                    self.nodetype[node] = builtins_ty[x]
                    return self.nodetype[node]
                raise self.ArgumentRequired(func=x)

            return self.nodetype[node]

        assert False


    # ================================= slice ==================================
    def infer_slice(self, node, ty_key_expected) -> 'NoneType':
        if isinstance(node, gast.Slice):
            # Slice(expr? lower, expr? upper, expr? step)
            if node.lower:
                ty_lower = self.infer_expr(node.lower)
                unify(ty_lower, ty_key_expected)
            if node.upper:
                ty_upper = self.infer_expr(node.upper)
                unify(ty_upper, ty_key_expected)
            if node.step:
                ty_step = self.infer_expr(node.step)
                unify(ty_step, ty_key_expected)
            return

        if isinstance(node, gast.Index):
            # Index(expr value)
            ty_val = self.infer_expr(node.value)
            unify(ty_val, ty_key_expected)
            return



if __name__ == '__main__':
    from copy import deepcopy
    import ast
    import gast
    import importlib
    import sys
    import traceback

    try:
        from astmonkey import transformers, visitors
        IMPORT_ASTMONKEY = True
    except ImportError:
        IMPORT_ASTMONKEY = False

    def dump_ast(mod, name):
        if IMPORT_ASTMONKEY:
            mod = deepcopy(mod)
            mod = transformers.ParentChildNodeTransformer().visit(deepcopy(mod))
            visitor = visitors.GraphNodeVisitor()
            visitor.visit(mod)
            visitor.graph.write_png(name + '.png')
            print("\033[1;32;40mAST visualization saved as \033[94m%s.png\033[0m" % name)
        else:
            print("\033[93mInstall astmonkey for visualization.\033[0m")

    if len(sys.argv) == 3:
        module = importlib.import_module(sys.argv[1])
        func = getattr(module, sys.argv[2])
        code = utils.clip_head(inspect.getsource(func))
    else:
        module = None
        code = open(sys.argv[1]).read()
    orig_ast = gast.ast_to_gast(ast.parse(code))
    dump_ast(orig_ast, 'original')

    is_debug_global = True
    tc = TypeChecker(is_debug=True, module=module)
    try:
        nodetype = tc.infer(orig_ast)
    except UnifyError as e:
        print(traceback.format_exc(), end="")
