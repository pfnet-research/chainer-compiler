import ast
import inspect
import gast
import os
import sys
import traceback
import types
from copy import deepcopy
from typing import List
from pprint import pprint

from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser.types import *

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import logging

# ==============================================================================

def debug(sth):
    frame = inspect.currentframe().f_back
    print("[{} {}] {}".format(frame.f_code.co_name, frame.f_lineno, sth))


def add_dict(dest, src):
    for k, v in src.items():
        dest[k] = v

def find(seq, pred):
    for elt in seq:
        if pred(elt):
            return elt


def copy_ty(ty):
    if isinstance(ty, TyUserDefinedClass):
        # XXX: do not copy instance
        ret = TyUserDefinedClass(ty.name, ty.instance)
    elif isinstance(ty, TyNum):
        ret = TyNum(kind=ty.kind, value=None)
    elif isinstance(ty, TyString):
        ret = TyString()
    elif isinstance(ty, TyDType):
        ret = TyDType()
    else:
        ret = deepcopy(ty)
    ret.is_optional = ty.is_optional
    return ret


def copy_tyenv(tyenv):
    new_tyenv = {}
    for name, ty in tyenv.items():
        new_tyenv[name] = copy_ty(ty)
    return new_tyenv


def copy_TypeChecker(tc):
    new_tc = TypeChecker(
            tyenv=tc.tyenv, attribute_tyenv=tc.attribute_tyenv,
            is_debug=tc.is_debug, module=tc.module)
    return new_tc


def lazy_initializer(node):
    def ident_eq(expr1, expr2):
        if isinstance(expr1, gast.Name) and isinstance(expr2, gast.Name):
            return expr1.id == expr2.id
        if isinstance(expr1, gast.Attribute) and isinstance(expr2, gast.Attribute):
            return ident_eq(expr1.value, expr2.value) and \
                    expr1.attr == expr2.attr

    # XXX: lazy initialization must be written as follows:
    #   if x is None: ...  (else: ...)
    #
    # The reverse, 'if x is not None: ... else: ...' is not supported.
    if isinstance(node.test, gast.Compare) and \
            (isinstance(node.test.left, gast.Name) or \
            isinstance(node.test.left, gast.Attribute)) and \
            isinstance(node.test.ops[0], gast.Is) and \
            isinstance(node.test.comparators[0], gast.NameConstant) and \
            node.test.comparators[0].value is None:
        x = node.test.left  # variable/attribute being initialized
        assign_x = [isinstance(stmt, gast.Assign) and \
                ident_eq(stmt.targets[0], x) for stmt in node.body]
        if any(assign_x):
            return node.test.left
    return None


def get_kwarg(ty_kwarg, key, default=None):
    return value_of_type(ty_kwarg[key]) if key in ty_kwarg.keys() \
            else default


def call_ext_function(func, node, ty_args, ty_kwargs):
    # Non-tensor arguments
    val_dummy_args_nontensor = [value_of_type(t) for t in ty_args \
            if not isinstance(t, TyTensor)]
    inference_logic = ext_func_ty[func]
    try:
        ty_ret = inference_logic(
                ty_args, val_dummy_args_nontensor, ty_kwargs)
    except Exception:
        print_warning("Failed to infer type of " + func.__name__ +
                ". Falling back to TyObj...")
        ty_ret = TyObj(lineno=getattr(node, 'lineno', None))
        # raise Exception
    return ty_ret


def call_ext_callable(obj, node, ty_args, ty_kwargs):
    inference_logic = ext_callable_ty[type(obj)]
    ty_ret = inference_logic(obj, ty_args, ty_kwargs)
    return ty_ret


def call_builtin_function(func, node, ty_args):
    dummy_args = [value_of_type(t) for t in ty_args]
    try:
        ty_ret = type_of_value(func(*dummy_args))
    except Exception:
        print_warning("Failed to infer type of " + func.__name__ +
                ". Falling back to TyObj...")
        ty_ret = TyObj(lineno=getattr(node, 'lineno', None))
        raise Exception
    return ty_ret


def call_binop(op, node, tyl, tyr):
    semantics = {
            gast.Add : (lambda x, y: x + y),
            gast.Sub : (lambda x, y: x - y),
            gast.Mult : (lambda x, y: x * y),
            gast.Div : (lambda x, y: x / y),
            gast.FloorDiv : (lambda x, y: x // y),
            }
    func = semantics[type(op)]
    vall, valr = value_of_type(tyl), value_of_type(tyr)
    try:
        ty_ret = type_of_value(func(vall, valr))
    except Exception:
        print_warning("Failed to infer type of " + func.__name__ +
                ". Falling back to TyObj...")
        ty_ret = TyObj(lineno=getattr(node, 'lineno', None))

    if isinstance(ty_ret, TySequence) and \
            not (tyl.is_fixed_len and tyr.is_fixed_len):
        ty_ret.coerce_to_variable_len()

    if isinstance(ty_ret, TyTensor) and \
            isinstance(tyl, TyTensor) and tyl.shape is None or \
            isinstance(tyr, TyTensor) and tyr.shape is None:
        ty_ret.shape = None
    return ty_ret


def choose_stronger_ty(ty1, ty2):
    if isinstance(ty1, TyNone):
        return ty2
    if isinstance(ty2, TyNone):
        return ty1
    if type(ty1) is TyObj:
        return ty2
    if type(ty2) is TyObj:
        return ty1
    return ty1  # whichever is okay


# ==============================================================================

# TODO
func_to_ignore = [print, logging.info]


def make_infer(func, fallback_shapes, fallback_dtypes):
    def infer(ty_args, dummy_args_nontensor, ty_kwargs):
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
        dummy_kwargs = {k : value_of_type(t) for (k, t) in ty_kwargs.items()}
        dummy_result = func(*dummy_args, **dummy_kwargs)
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
    def infer(ty_args, dummy_args_nontensor, ty_kwargs):
        ksize = dummy_args_nontensor[0]
        stride = get_kwarg(ty_kwargs, 'stride', default=ksize)
        minimum_size = max(ksize, stride)
        fallback_shapes = ((1, 1, minimum_size, minimum_size),)
        fallback_dtypes = (np.float32,)

        return make_infer(func, fallback_shapes, fallback_dtypes) \
                (ty_args, dummy_args_nontensor, ty_kwargs)

    return infer


def ty_ChainerSoftmaxCrossEntropy(ty_args, dummy_args_nontensor, ty_kwargs):
    shape_x, shape_t = ty_args[0].shape, ty_args[1].shape
    fallback_dtypes = (np.float32, np.int64)

    # x.shape[0] == t.shape[0]
    if shape_x is None and shape_t is not None:
        fallback_shapes = ((shape_t[0], 1), shape_t)
    elif shape_x is not None and shape_t is None:
        fallback_shapes = (shape_x, (shape_x[0],))
    else:
        fallback_shapes = ((1, 1), (1,))

    return make_infer(
            F.softmax_cross_entropy, fallback_shapes, fallback_dtypes) \
                    (ty_args, dummy_args_nontensor, ty_kwargs)


# math functions that doesn't change shapes or dtypes
def ty_ChainerIdentical(is_float_only=True):
    def infer(ty_args, dummy_args_nontensor, ty_kwargs):
        if isinstance(ty_args[0], TyTensor):
            if is_float_only:
                assert ty_args[0].dtype.is_float()
            return ty_args[0]
        assert False

    return infer


def ty_ChainerConcat(ty_args, dummy_args_nontensor, ty_kwargs):
    # TODO(momohatt): shape
    assert isinstance(ty_args[0], TySequence)
    if ty_args[0].is_fixed_len:
        dtypes = [tytensor.dtype for tytensor in ty_args[0].get_tys()]
        assert all_same(dtypes)
        return TyChainerVariable(dtype=dtypes[0])

    dtype = ty_args[0].get_ty().dtype
    return TyChainerVariable(dtype=dtype)


def ty_ChainerExpandDims(ty_args, dummy_args_nontensor, ty_kwargs):
    # TODO(momohatt): axis can come as dummy_args_nontensor
    axis = dummy_args_nontensor[0]
    fallback_shapes = ((1,) * axis,)
    fallback_dtypes = (np.float32,)

    return make_infer(F.expand_dims, fallback_shapes, fallback_dtypes) \
            (ty_args, dummy_args_nontensor, ty_kwargs)


def ty_ChainerBroadcastTo(ty_args, dummy_args_nontensor, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSum(ty_args, dummy_args_nontensor, ty_kwargs):
    axis = get_kwarg(ty_kwargs, 'axis', default=None)
    fallback_shapes = ((1,) * (axis + 1),)
    fallback_dtypes = (np.float32,)

    return make_infer(F.sum, fallback_shapes, fallback_dtypes) \
            (ty_args, dummy_args_nontensor, ty_kwargs)


def ty_ChainerReshape(ty_args, dummy_args_nontensor, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype,
            shape=dummy_args_nontensor[0])


def ty_ChainerSqueeze(ty_args, dummy_args_nontensor, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSwapAxes(ty_args, dummy_args_nontensor, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSeparate(ty_args, dummy_args_nontensor, ty_kwargs):
    return TyChainerVariable(dtype=ty_args[0].dtype)


def ty_ChainerSplitAxis(ty_args, dummy_args_nontensor, ty_kwargs):
    assert isinstance(ty_args[0], TyTensor)

    if isinstance(ty_args[1], TyNum):
        n = dummy_args_nontensor[0]
        return TyTuple([TyChainerVariable(dtype=ty_args[0].dtype)] * n)
    elif isinstance(ty_args[1], TyTensor):
        # 1-D array
        if ty_args[1].shape is None:
            # variable length tuple
            return TyTuple(TyChainerVariable(dtype=ty_args[0].dtype))
        n = ty_args[1].shape[0]
        return TyTuple([TyChainerVariable(dtype=ty_args[0].dtype)] * n)

    assert False


def ty_ChainerPadSequence(ty_args, dummy_args_nontensor, ty_kwargs):
    # shape[1:] should be uniform & ndim > 0
    ty = ty_args[0].deref()
    assert isinstance(ty, TySequence)
    is_dummy_shape = False
    if ty.is_fixed_len:
        is_shape_None = [t.shape is None for t in ty.get_tys()]
        if any(is_shape_None):
            is_dummy_shape = True
            if all(is_shape_None):
                dummy_args_nontensor[0] = [np.zeros((1,)) for _ in ty.get_tys()]
            else:
                t = find(ty.get_tys(), lambda t: t.shape is not None)
                fallback_shape = t.shape
                dummy_args_nontensor[0] = [
                        np.zeros(fallback_shape) if t.shape is None
                        else np.zeros(t.shape) for t in ty.get_tys()]
    else:
        if ty.get_ty().shape is None:
            is_dummy_shape = True
            dummy_args_nontensor[0] = [np.zeros((1,), dtype=ty.get_ty().dtype.t)]

    dummy_kwargs = {k : value_of_type(t) for (k, t) in ty_kwargs.items()}
    ty_ret = type_of_value(F.pad_sequence(*dummy_args_nontensor, **dummy_kwargs))
    if is_dummy_shape:
        ty_ret.shape = None
    return ty_ret


def ty_ChainerLinear(obj, ty_args, ty_kwargs):
    shape = ty_args[0].shape
    dtype = ty_args[0].dtype

    if obj.in_size is not None and shape is not None:
        if len(shape) > 2:
            # case of reshape
            pass
        assert len(shape) == 2
        assert shape[1] == obj.in_size
    assert dtype.t == obj.b.dtype
    return TyChainerVariable(dtype=dtype, shape=(shape[0], obj.out_size))


ext_func_ty = {
        np.array : evaluate_function_types(
            np.array, 0),
        np.cumsum :
            ty_ChainerIdentical(is_float_only=False),
        np.full : evaluate_function_types(
            np.full, 0),
        np.ones : evaluate_function_types(
            np.ones, 0),
        np.zeros : evaluate_function_types(
            np.zeros, 0),
        chainer.Variable : evaluate_function_types(
            chainer.Variable, 1),
        cuda.to_cpu :
            ty_ChainerIdentical(is_float_only=False),
        F.average_pooling_2d :
            ty_ChainerPooling2d(F.average_pooling_2d),
        F.broadcast_to :
            ty_ChainerBroadcastTo,
        F.concat :
            ty_ChainerConcat,
        F.dropout :
            ty_ChainerIdentical(),
        F.expand_dims :
            ty_ChainerExpandDims,
        F.hstack :
            ty_ChainerConcat,
        F.local_response_normalization : evaluate_function_types(
            F.local_response_normalization, 1),
        F.max_pooling_2d :
            ty_ChainerPooling2d(F.max_pooling_2d),
        F.pad_sequence :
            ty_ChainerPadSequence,
        F.relu :
            ty_ChainerIdentical(),
        F.reshape :
            ty_ChainerReshape,
        F.separate :
            ty_ChainerSeparate,
        F.sigmoid :
            ty_ChainerIdentical(),
        F.split_axis :
            ty_ChainerSplitAxis,
        F.squeeze :
            ty_ChainerSqueeze,
        F.softmax :
            ty_ChainerIdentical(),
        F.softmax_cross_entropy :
            ty_ChainerSoftmaxCrossEntropy,
        F.stack :
            ty_ChainerConcat,
        F.sum :
            ty_ChainerSum,
        F.swapaxes :
            ty_ChainerSwapAxes,
        F.tanh :
            ty_ChainerIdentical(),
        F.vstack :
            ty_ChainerConcat,
        }


ext_callable_ty = {
        L.Linear :
            ty_ChainerLinear,
        }


list_attr_ty = {
        'append'  : lambda x: TyArrow([x.get_ty()], TyNone()),
        'reverse' : lambda x: TyArrow([], TyNone()),
        }


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

        # Path from the root of AST to the current node (only stmt and expr)
        self.stack = []


    def dump_tyenv(self):
        if not self.is_debug:
            return
        print("=== tyenv ===")
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
            print("{} : \x1b[36m{}\x1b[39m".format(
                utils.node_description(node), ty))
        print()


    def dump_stack(self):
        print("=== stack ===\x1b[32m")
        for node in self.stack:
            print(utils.node_description(node))
        print("\x1b[39m")


    def is_called(self, node):
        if len(self.stack) < 2:
            return False
        return isinstance(self.stack[-2], gast.Call) and \
                self.stack[-2].func is node


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

        if isinstance(node, gast.Name) and node.id in self.tyenv.keys() and \
                isinstance(self.tyenv[node.id], TyUserDefinedClass):
            # ex. value of 'self'
            return self.tyenv[node.id].instance


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

        if self.is_debug:
            print("\x1b[33m======================= function {} =======================\x1b[39m".format(node.name))

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


    def infer_block(self, tc, stmts):  # use in if (without else), for, while
        for stmt in stmts:
            tc.infer_stmt(stmt)

        # unify the intersection of 2 tyenvs and update local tyenv
        for name, ty in tc.tyenv.items():
            if name in self.tyenv.keys():
                unify(ty, self.tyenv[name])
            self.tyenv[name] = ty

        for (obj, name), ty in tc.attribute_tyenv.items():
            if (obj, name) in self.attribute_tyenv.keys():
                unify(ty, self.attribute_tyenv[(obj, name)])
            self.attribute_tyenv[(obj, name)] = ty


    def infer_2blocks(self, tc1, tc2, stmts1, stmts2):
        for stmt in stmts1:
            tc1.infer_stmt(stmt)
        for stmt in stmts2:
            tc2.infer_stmt(stmt)

        # unify the intersection of 2 tyenvs and update local tyenv
        for name, ty in tc1.tyenv.items():
            if name in tc2.tyenv.keys():
                unify(ty, tc2.tyenv[name])
                self.tyenv[name] = choose_stronger_ty(ty, tc2.tyenv[name])
            else:
                self.tyenv[name] = ty
        for name, ty in tc2.tyenv.items():
            if name in tc1.tyenv.keys():
                continue
            self.tyenv[name] = ty

        for (obj, name), ty in tc1.attribute_tyenv.items():
            if (obj, name) in tc2.attribute_tyenv.keys():
                unify(ty, tc2.attribute_tyenv[(obj, name)])
                self.attribute_tyenv[(obj, name)] = \
                        choose_stronger_ty(ty, tc2.attribute_tyenv[(obj, name)])
            else:
                self.attribute_tyenv[(obj, name)] = ty
        for (obj, name), ty in tc2.attribute_tyenv.items():
            if (obj, name) in tc1.attribute_tyenv.keys():
                continue
            self.attribute_tyenv[(obj, name)] = ty


    def infer_user_defined_function(self, func, ty_args, node):
        if isinstance(func, types.FunctionType) or \
                isinstance(func, types.MethodType):
            func_body = func

            if isinstance(node.func, gast.Attribute):
                ty_self = self.nodetype[node.func.value]
                ty_args = [ty_self] + ty_args

        else:
            # defined with __call__
            if isinstance(func, chainer.Chain):
                func_body = func.forward
            else:
                func_body = func.__call__

            ty_self = type_of_value(func)
            ty_args = [ty_self] + ty_args

        code = utils.clip_head(inspect.getsource(func_body))
        # FunctionDef of called subroutine
        func_node = gast.ast_to_gast(ast.parse(code)).body[0]
        self.subroutine_node[node] = func_node
        tc = TypeChecker(is_debug=self.is_debug,
                module=sys.modules[func.__module__])
        tc.infer_function(func_node, ty_args)

        # copy nodetype and subroutine_node from subroutine
        add_dict(self.nodetype, tc.nodetype)
        add_dict(self.subroutine_node, tc.subroutine_node)
        return ty_args, tc.nodetype[func_node]


    # ================================ mod =====================================
    def infer_mod(self, node):
        if isinstance(node, gast.Module):
            self.infer_stmt(node.body[0])
            return


    # ================================ stmt ====================================
    def infer_stmt(self, node) -> 'TyObj':
        if self.is_debug:
            debug(gast.dump(node))

        self.stack.append(node)

        if isinstance(node, gast.FunctionDef):
            self.nodetype[node] = self.infer_FunctionDef(node)
        elif isinstance(node, gast.Return):
            # Return(expr? value)
            if node.value is None:
                self.nodetype[node] = TyNone()
            else:
                self.nodetype[node] = self.infer_expr(node.value)
        elif isinstance(node, gast.Delete):
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.Assign):
            self.infer_Assign(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.AugAssign):
            self.infer_AugAssign(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.For):
            self.infer_For(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.While):
            # While(expr test, stmt* body, stmt* orelse)
            pass
        elif isinstance(node, gast.If):
            self.infer_If(node)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.Expr):
            # Expr(expr value)
            self.infer_expr(node.value)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.Pass):
            self.nodetype[node] = TyNone()

        assert node in self.nodetype.keys(), type(node).__name__
        self.stack.pop()
        return self.nodetype[node]


    def infer_FunctionDef(self, node):
        # FunctionDef(identifier name, arguments args, stmt* body,
        # expr* decorator_list, expr? returns)
        ty_args = [self.tyenv[arg.id] for arg in node.args.args]
        ty = None

        for stmt in node.body:
            ty = self.infer_stmt(stmt)

        assert ty is not None
        return TyArrow(ty_args, ty)


    def infer_Assign(self, node):
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
                return
            self.tyenv[target.id] = copy_ty(ty_val)
            self.nodetype[target] = copy_ty(ty_val)
            return

        if isinstance(target, gast.Attribute):
            self.infer_expr(target)
            ty_obj = self.nodetype[target.value]
            assert isinstance(ty_obj, TyUserDefinedClass)
            self.attribute_tyenv[(ty_obj.instance, target.attr)] = ty_val
            return

        if type(target) in [gast.Tuple, gast.List]:
            ty_target = self.infer_expr(target)
            unify(ty_target, ty_val)
            for (var, ty) in zip(target.elts, ty_val.get_tys()):
                self.tyenv[var.id] = ty
                self.nodetype[var] = ty
            return


    def infer_AugAssign(self, node):
        # AugAssign(expr target, operator op, expr value)
        binop = gast.BinOp(node.target, node.op, node.value)
        if hasattr(node, 'lineno'):
            setattr(binop, 'lineno', node.lineno)
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


    def infer_For(self, node):
        # For(expr target, expr iter, stmt* body, stmt* orelse)
        assert type(node.target) in [gast.Name, gast.Tuple]

        ty_iteration = self.infer_expr(node.iter)
        ty_i = self.infer_expr(node.target)
        if isinstance(ty_iteration, TyTensor):
            unify(ty_i, TyTensor(
                dtype=ty_iteration.dtype, kind=ty_iteration.kind))
        else:
            unify(ty_iteration, TySequence(ty=ty_i))

        for _ in range(2):
            tc = copy_TypeChecker(self)
            self.infer_block(tc, node.body)

        add_dict(self.nodetype, tc.nodetype)
        add_dict(self.subroutine_node, tc.subroutine_node)


    def infer_If(self, node):
        # If(expr test, stmt* body, stmt* orelse)
        # TODO(momohatt): determine what type should ty_test be
        self.infer_expr(node.test)
        x = lazy_initializer(node)

        if node.orelse == []:
            tc = copy_TypeChecker(self)
            self.infer_block(tc, node.body)
            add_dict(self.nodetype, tc.nodetype)
            add_dict(self.subroutine_node, tc.subroutine_node)
        else:
            tc1 = copy_TypeChecker(self)
            tc2 = copy_TypeChecker(self)
            self.infer_2blocks(tc1, tc2, node.body, node.orelse)
            add_dict(self.nodetype, tc1.nodetype)
            add_dict(self.nodetype, tc2.nodetype)
            add_dict(self.subroutine_node, tc1.subroutine_node)
            add_dict(self.subroutine_node, tc2.subroutine_node)

        if x is not None:
            self.infer_expr(x).is_optional = False


    # ================================= expr ===================================
    def infer_expr(self, node) -> 'TyObj':
        if node in self.nodetype.keys():
            return self.nodetype[node]

        self.stack.append(node)

        if self.is_debug:
            debug(gast.dump(node))
            # self.dump_stack()
            # self.dump_tyenv()

        if isinstance(node, gast.BoolOp):
            self.nodetype[node] = self.infer_BoolOp(node)
        elif isinstance(node, gast.BinOp):
            self.nodetype[node] = self.infer_BinOp(node)
        elif isinstance(node, gast.UnaryOp):
            self.nodetype[node] = self.infer_UnaryOp(node)
        elif isinstance(node, gast.Dict):
            self.nodetype[node] = self.infer_Dict(node)
        elif isinstance(node, gast.ListComp):
            self.nodetype[node] = self.infer_ListComp(node)
        elif isinstance(node, gast.Compare):
            # Compare(expr left, cmpop* ops, expr* comparators)
            self.nodetype[node] = TyBool()
        elif isinstance(node, gast.Call):
            self.nodetype[node] = self.infer_Call(node)
        elif isinstance(node, gast.Num):
            # Num(object n)
            self.nodetype[node] = type_of_value(node.n)
        elif isinstance(node, gast.Str):
            # Str(string s)
            self.nodetype[node] = TyString(value=node.s)
        elif isinstance(node, gast.NameConstant):
            # NameConstant(singleton value)
            self.nodetype[node] = type_of_value(node.value)
        elif isinstance(node, gast.Attribute):
            self.nodetype[node] = self.infer_Attribute(node)
        elif isinstance(node, gast.Subscript):
            self.nodetype[node] = self.infer_Subscript(node)
        elif isinstance(node, gast.Name):
            self.nodetype[node] = self.infer_Name(node)
        elif isinstance(node, gast.List):
            # List(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyList(elts_ty)
        elif isinstance(node, gast.Tuple):
            # Tuple(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyTuple(elts_ty)

        assert node in self.nodetype.keys() and \
                self.nodetype[node] is not None, type(node).__name__
        self.stack.pop()
        return self.nodetype[node]


    def infer_BoolOp(self, node):
        # BoolOp(boolop op, expr* values)
        ty_vals = [self.infer_expr(val) for val in node.values]
        for ty in ty_vals:
            unify(ty, TyBool())
        self.nodetype[node.op] = TyArrow([TyBool(), TyBool()], TyBool())
        return TyBool()


    def infer_BinOp(self, node):
        # BinOp(expr left, operator op, expr right)
        tyl = self.infer_expr(node.left).deref()
        tyr = self.infer_expr(node.right).deref()
        ty_ret = call_binop(node.op, node, tyl, tyr)
        self.nodetype[node.op] = TyArrow([tyl, tyr], ty_ret)
        return ty_ret


    def infer_UnaryOp(self, node):
        # UnaryOp(unaryop op, expr operand)
        if isinstance(node.op, gast.Invert):
            pass
        elif isinstance(node.op, gast.Not):
            pass
        elif isinstance(node.op, gast.UAdd):
            pass
        elif isinstance(node.op, gast.USub):
            ty_expr = self.infer_expr(node.operand)
            if isinstance(ty_expr, TyNum) and ty_expr.value is not None:
                return type_of_value(- ty_expr.value)
            return ty_expr


    def infer_Dict(self, node):
        # Dict(expr* keys, expr* values)
        if node.keys == []:
            return TyDict(TyVar(), TyVar())
        ty_keys = [self.infer_expr(key) for key in node.keys]
        ty_vals = [self.infer_expr(val) for val in node.values]
        assert all_same_ty(ty_keys)
        assert all_same_ty(ty_vals)
        return TyDict(ty_keys[0], ty_vals[0])


    def infer_ListComp(self, node):
        # ListComp(expr elt, comprehension* generators)

        # cannot think of cases where len > 2
        assert len(node.generators) == 1

        tc = copy_TypeChecker(self)
        gen = node.generators[0]
        ty_iteration = tc.infer_expr(gen.iter)
        ty_i = tc.infer_expr(gen.target)
        if isinstance(ty_iteration, TyTensor):
            unify(ty_i, TyTensor(
                dtype=ty_iteration.dtype, kind=ty_iteration.kind))
        else:
            unify(ty_iteration, TySequence(ty=ty_i))
        tc.infer_expr(node.elt)

        add_dict(self.nodetype, tc.nodetype)
        add_dict(self.subroutine_node, tc.subroutine_node)

        self.nodetype[node] = TyList(tc.nodetype[node.elt])
        return self.nodetype[node]


    def infer_Call(self, node):
        # Call(expr func, expr* args, keyword* keywords)
        ty_args = [self.infer_expr(arg) for arg in node.args]
        ty_kwargs = {kwarg.arg : self.infer_expr(kwarg.value) \
                for kwarg in node.keywords}
        ty_ret = TyVar()

        try:
            ty_fun = self.infer_expr(node.func)
        except self.ArgumentRequired as e:
            self.stack.pop()  # for node.func

            if e.func in func_to_ignore:
                return TyNone()

            if e.func is None:
                # attribute against tensor etc.
                assert isinstance(node.func, gast.Attribute)
                ty_obj = self.nodetype[node.func.value]

                if isinstance(ty_obj, TyTensor) and ty_obj.is_ndarray():
                    if node.func.attr == 'astype':
                        val_args = [self.evaluate(arg) for arg in node.args]
                        ty_ret = TyNdarray(dtype=TyDType(val_args[0]),
                                shape=ty_obj.shape)

                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
                return ty_ret

            if e.func in ext_func_ty.keys():
                # external (eg. np/chainer) functions
                ty_ret = call_ext_function(e.func, node, ty_args, ty_kwargs)
                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
                return ty_ret

            if e.func in __builtins__.values():
                # builtin functions
                ty_ret = call_builtin_function(e.func, node, ty_args)
                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
                return ty_ret

            if type(e.func) in L.__dict__.values():
                # chainer links
                ty_ret = call_ext_callable(e.func, node, ty_args, ty_kwargs)
                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
                return ty_ret

            # user defined functions/methods/callables, need to inline
            ty_args, ty_fun = self.infer_user_defined_function(
                    e.func, ty_args, node)

        unify(ty_fun, TyArrow(ty_args, ty_ret))
        self.nodetype[node.func] = ty_fun
        return ty_ret.deref()


    def infer_Attribute(self, node):
        # Attribute(expr value, identifier attr, expr_context ctx)

        if isinstance(node.value, gast.Name) and \
                hasattr(self.module, node.value.id):
            # function of imported libraries (eg. np, chainer, F, L)
            module = getattr(self.module, node.value.id)
            attr = getattr(module, node.attr)
            if self.is_called(node):
                raise self.ArgumentRequired(func=attr)
            return type_of_value(attr)

        ty_obj = self.infer_expr(node.value).deref()

        if isinstance(ty_obj, TySequence) and ty_obj.is_list():
            ty_obj.coerce_to_variable_len()
            return list_attr_ty[node.attr](ty_obj)

        if isinstance(ty_obj, TyTensor):
            if node.attr == 'shape':
                if ty_obj.shape is None:
                    return TyTuple(TyInt())
                return type_of_value(ty_obj.shape)
            if node.attr == 'size':
                return TyInt()
            if ty_obj.is_ndarray() and node.attr == 'astype':
                raise self.ArgumentRequired()
            assert False

        if isinstance(ty_obj, TyUserDefinedClass):
            # x: value of existing instance
            x = getattr(ty_obj.instance, node.attr)

            if callable(x) and x in __builtins__.keys():
                raise self.ArgumentRequired(func=x)

            if (ty_obj.instance, node.attr) in self.attribute_tyenv.keys():
                ty_node = self.attribute_tyenv[(ty_obj.instance, node.attr)]
            else:
                ty_node = type_of_value(x)

            if self.is_called(node):
                raise self.ArgumentRequired(func=x)

            return ty_node


    def infer_Subscript(self, node):
        # Subscript(expr value, slice slice, expr_context ctx)
        ty_obj = self.infer_expr(node.value)

        if isinstance(ty_obj, TySequence):
            self.infer_slice(node.slice, TyInt())
            if ty_obj.is_fixed_len and \
                    isinstance(node.slice, gast.Index) and \
                    isinstance(node.slice.value, gast.Num):
                return ty_obj.get_tys()[node.slice.value.n]

            ty_obj.coerce_to_variable_len()
            if isinstance(node.slice, gast.Index):
                return ty_obj.get_ty()
            if isinstance(node.slice, gast.Slice):
                return ty_obj
            assert False, "indices must be integers or slices"

        if isinstance(ty_obj, TyDict):
            self.infer_slice(node.slice, ty_obj.keyty)
            assert isinstance(node.slice, gast.Index)
            return ty_obj.valty

        if isinstance(ty_obj, TyTensor):
            self.infer_slice(node.slice, TyInt())
            return TyTensor(dtype=ty_obj.dtype, kind=ty_obj.kind)


    def infer_Name(self, node):
        # Name(identifier id, expr_context ctx, expr? annotation)
        if node.id in self.tyenv.keys():
            return self.tyenv[node.id]
        if node.id in __builtins__.keys():
            value = __builtins__[node.id]
            if callable(value) and self.is_called(node):
                raise self.ArgumentRequired(func=value)
            return type_of_value(value)
        if hasattr(self.module, node.id):
            x = getattr(self.module, node.id)
            if self.is_called(node):
                raise self.ArgumentRequired(func=x)
            return type_of_value(x)

        # case of Tuple assignment
        # XXX: print comes here
        ty_var = TyVar()
        self.tyenv[node.id] = ty_var
        return ty_var


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

    tc = TypeChecker(is_debug=True, module=module)
    try:
        nodetype = tc.infer(orig_ast)
    except UnifyError as e:
        print(traceback.format_exc(), end="")
