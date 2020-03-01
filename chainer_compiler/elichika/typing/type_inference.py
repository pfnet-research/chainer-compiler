import ast
import collections
import inspect
import gast
import numbers
import sys
import types
import typing

from   chainer_compiler.elichika.parser.utils             import clip_head
from   chainer_compiler.elichika.typing.ext.common        import ty_TensorArith
from   chainer_compiler.elichika.typing.ext.numpy_functions   import numpy_func_ty
from   chainer_compiler.elichika.typing.ext.chainer_functions import chainer_func_ty, chainer_callable_ty
from   chainer_compiler.elichika.typing.ext.pytorch_functions import pytorch_func_ty, pytorch_callable_ty
from   chainer_compiler.elichika.typing.types             import *
from   chainer_compiler.elichika.typing.shape_elem        import *
from   chainer_compiler.elichika.typing                   import utils

import chainer
from   chainer.backends import cuda
import chainer.links as L
import numpy as np
import logging

import torch
import torch.nn as nn

# ==============================================================================

def debug(sth):
    frame = inspect.currentframe().f_back
    print("[{} {}] {}".format(frame.f_code.co_name, frame.f_lineno, sth))


def copy_tyenv(tyenv):
    new_tyenv = {}
    for name, ty in tyenv.items():
        new_tyenv[name] = copy_ty(ty)
    return new_tyenv


def copy_InferenceEngine(tc):
    new_tc = InferenceEngine(
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
            isinstance(node.test.left, (gast.Name, gast.Attribute)) and \
            isinstance(node.test.ops[0], gast.Is) and \
            isinstance(node.test.comparators[0], gast.Constant) and \
            node.test.comparators[0].value is None:
        x = node.test.left  # variable/attribute being initialized
        assign_x = [isinstance(stmt, gast.Assign) and \
                ident_eq(stmt.targets[0], x) for stmt in node.body]
        if any(assign_x):
            return node.test.left
    return None


def handle_inference_error(exception, func, node):
    if hasattr(func, '__class__'):
        name = func.__class__.__name__
    elif hasattr(func, '__name__'):
        name = func.__name__
    else:
        name = str(func)
    utils.print_warning(str(exception))
    utils.print_warning("Failed to infer type of " + name +
            ". Falling back to TyVar...")
    # raise Exception
    return TyVar(lineno=getattr(node, 'lineno', None))


def call_ext_function(table, func, node, ty_args, ty_kwargs):
    inference_logic = table[func]
    try:
        ty_ret = inference_logic(ty_args, ty_kwargs)
    except Exception as e:
        ty_ret = handle_inference_error(e, func, node)
    return ty_ret


def call_ext_callable(table, obj, node, ty_args, ty_kwargs):
    inference_logic = table[type(obj)]
    try:
        ty_ret = inference_logic(obj, ty_args, ty_kwargs)
    except Exception as e:
        ty_ret = handle_inference_error(e, obj, node)
    return ty_ret


def call_builtin_function(func, node, ty_args):
    try:
        dummy_args = [generate_dummy_value(t) for t in ty_args]
        ty_ret = type_of_value(func(*dummy_args))
    except Exception as e:
        ty_ret = handle_inference_error(e, func, node)
    return ty_ret


def call_binop(op, node, tyl, tyr):
    if isinstance(tyl, TyTensor):
        return ty_TensorArith(tyl.kind)([tyl, tyr], {})
    if isinstance(tyr, TyTensor):
        return ty_TensorArith(tyr.kind)([tyl, tyr], {})

    semantics = {
            gast.Add : (lambda x, y: x + y),
            gast.Sub : (lambda x, y: x - y),
            gast.Mult : (lambda x, y: x * y),
            gast.Div : (lambda x, y: x / y),
            gast.FloorDiv : (lambda x, y: x // y),
            }
    func = semantics[type(op)]
    try:
        vall, valr = generate_dummy_value(tyl), generate_dummy_value(tyr)
        ty_ret = type_of_value(func(vall, valr))
    except Exception as e:
        ty_ret = handle_inference_error(e, op.__class__.__name__, node)

    if isinstance(ty_ret, TySequence) and \
            not (tyl.is_fixed_len and tyr.is_fixed_len):
        ty_ret.coerce_to_variable_len()

    return ty_ret


# ==============================================================================

func_to_ignore = [logging.info]


list_attr_ty = {
        'append'  : lambda x: TyArrow([x.get_ty()], TyNone()),
        'reverse' : lambda x: TyArrow([], TyNone()),
        }

# ==============================================================================

class InferenceEngine():
    # TODO(momohatt): Don't use Exception
    class ArgumentRequired(Exception):
        def __init__(self, func=None, ty_obj=None):
            self.func = func  # callables
            self.ty_obj = ty_obj  # method call against

    def __init__(self, tyenv=None, attribute_tyenv=None, is_debug=False, module=None):
        # type environments for local objects
        # string -> TyObj
        self.tyenv = {} if tyenv is None else copy_tyenv(tyenv)

        # type environments for model attributes
        # (object, str) -> TyObj
        self.attribute_tyenv = {} if attribute_tyenv is None \
                else copy_tyenv(attribute_tyenv)

        # annotation to input AST
        # Node -> TyObj
        self.nodetype = {}

        self.is_debug = is_debug
        self.module = module

        # map from user-defined function call points to inlined function ASTs
        # Node (Call) -> Node (FunctionDef)
        self.subroutine_node = collections.OrderedDict()


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


    def dump_one_node(self, node):
        if node not in self.nodetype.keys():
            return
        print("{} : \x1b[36m{}\x1b[39m".format(
            utils.node_description(node), self.nodetype[node]))


    def generate_fresh_TyVar(self, node):
        assert isinstance(node, gast.Name)
        t = TyVar()
        self.nodetype[node] = t
        self.tyenv[node.id] = t
        return t


    def evaluate(self, node):
        if isinstance(node, gast.Attribute):
            v_value = self.evaluate(node.value)
            if v_value is None:
                return None
            attr = getattr(v_value, node.attr)
            return attr

        if isinstance(node, gast.Constant):
            return node.value

        if isinstance(node, gast.Name) and hasattr(self.module, node.id):
            return getattr(self.module, node.id)

        if isinstance(node, gast.Name) and node.id in self.tyenv.keys() and \
                isinstance(self.tyenv[node.id], TyUserDefinedClass):
            # ex. value of 'self'
            return self.tyenv[node.id].instance


    def infer(self, node):
        self.infer_mod(node)
        return self.nodetype


    def infer_function_value_args(self, node, args, type_hints={}):
        # args: argument value
        ty_args = [type_of_value(arg) for arg in args]
        return self.infer_function(node, ty_args, type_hints)


    def infer_function(self, node, ty_args, type_hints={}):
        # TODO(momohatt): varargs
        assert isinstance(node, gast.FunctionDef)
        if node.args.vararg is None:
            assert len(ty_args) == len(node.args.args), \
                    "Wrong number of arguments: expected {}, got {}".format(
                            len(node.args.args), len(ty_args))

        if self.is_debug:
            print("\x1b[33m==================== function {} ====================\x1b[39m".format(node.name))

        for arg_node, ty in zip(node.args.args, ty_args):
            self.tyenv[arg_node.id] = ty
        for ty in ty_args:
            if isinstance(ty, TyUserDefinedClass):
                for attr, val in ty.instance.__dict__.items():
                    self.attribute_tyenv[(ty.instance, attr)] = \
                            type_of_value(val)

        # apply type hints
        for n, t in type_hints.items():
            # TODO(momohatt): use term-match instead of unify?
            unify(self.tyenv[n], t)
            if isinstance(t, TyTensor):
                for i in range(self.tyenv[n].ndim):
                    self.tyenv[n].shape[i].expr = t.shape[i].expr

        self.infer_stmt(node)

        if self.is_debug:
            print('==================== Type Environment ====================')
            self.dump_nodetype()
        return self.nodetype


    def infer_block(self, tc, stmts):  # use in if (without else), for, while
        for stmt in stmts:
            ty_ret = tc.infer_stmt(stmt)

        # unify the intersection of 2 tyenvs and update local tyenv
        for name, ty in tc.tyenv.items():
            if name in self.tyenv.keys():
                unify(ty, self.tyenv[name])
            self.tyenv[name] = ty

        for (obj, name), ty in tc.attribute_tyenv.items():
            if (obj, name) in self.attribute_tyenv.keys():
                unify(ty, self.attribute_tyenv[(obj, name)])
            self.attribute_tyenv[(obj, name)] = ty

        unify(ty_ret, TyNone())
        return TyNone()


    def infer_2blocks(self, tc1, tc2, stmts1, stmts2):
        for stmt in stmts1:
            ty_ret1 = tc1.infer_stmt(stmt)
        for stmt in stmts2:
            ty_ret2 = tc2.infer_stmt(stmt)

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

        unify(ty_ret1, ty_ret2)
        return choose_stronger_ty(ty_ret1, ty_ret2)


    def infer_function_instance(self, node, func, ty_args, ty_kwargs):
        if func in numpy_func_ty.keys():
            return call_ext_function(numpy_func_ty, func, node, ty_args, ty_kwargs)

        if func in chainer_func_ty.keys():
            # external (eg. np/chainer) functions
            return call_ext_function(chainer_func_ty, func, node, ty_args, ty_kwargs)

        if func in pytorch_func_ty.keys():
            return call_ext_function(pytorch_func_ty, func, node, ty_args, ty_kwargs)

        if type(func) in L.__dict__.values():
            # chainer links
            return call_ext_callable(chainer_callable_ty, func, node, ty_args, ty_kwargs)

        if type(func) in nn.__dict__.values():
            # torch.nn
            if isinstance(func, nn.Sequential):
                x_type, = ty_args
                for idx, module in enumerate(func.children()):
                    x_type = self.infer_function_instance(node, module, [x_type], {})
                return x_type

            return call_ext_callable(pytorch_callable_ty, func, node, ty_args, ty_kwargs)

        if func in __builtins__.values():
            # builtin functions
            return call_builtin_function(func, node, ty_args)

        # user defined functions/methods/callables, need to inline
        return self.infer_user_defined_function(func, ty_args, node)


    def infer_user_defined_function(self, func, ty_args, node):
        if isinstance(func, (types.FunctionType, types.MethodType)):
            func_body = func

            if isinstance(node.func, gast.Attribute):
                ty_self = self.nodetype[node.func.value]
                ty_args = [ty_self] + ty_args

        else:
            # defined with __call__
            if isinstance(func, chainer.Chain) or isinstance(func, nn.Module):
                func_body = func.forward
            else:
                func_body = func.__call__

            ty_self = type_of_value(func)
            ty_args = [ty_self] + ty_args

        code = clip_head(inspect.getsource(func_body))
        # FunctionDef of called subroutine
        func_node = gast.ast_to_gast(ast.parse(code)).body[0]
        self.subroutine_node[node] = func_node
        tc = InferenceEngine(is_debug=self.is_debug,
                module=sys.modules[func.__module__])
        tc.infer_function(func_node, ty_args,
                type_hints=typing.get_type_hints(func_body))

        # copy nodetype and subroutine_node from subroutine
        utils.add_dict(self.nodetype, tc.nodetype)
        utils.add_dict(self.subroutine_node, tc.subroutine_node)
        return tc.nodetype[func_node].retty


    # ================================ mod =====================================
    def infer_mod(self, node):
        if isinstance(node, gast.Module):
            self.infer_stmt(node.body[0])
            return


    # ================================ stmt ====================================
    def infer_stmt(self, node):
        if self.is_debug:
            debug(gast.dump(node))

        if isinstance(node, gast.FunctionDef):
            self.nodetype[node] = self.infer_FunctionDef(node)
        elif isinstance(node, gast.Return):
            # Return(expr? value)
            if node.value is None:
                self.nodetype[node] = TyNone()
            else:
                self.nodetype[node] = self.infer_expr(node.value)
        elif isinstance(node, gast.Delete):
            # TODO(momohatt): erase from tyenv, etc.
            # TODO(momohatt): support deletion of element from list
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
            self.nodetype[node] = self.infer_If(node)
        elif isinstance(node, gast.Expr):
            # Expr(expr value)
            self.infer_expr(node.value)
            self.nodetype[node] = TyNone()
        elif isinstance(node, gast.Pass):
            self.nodetype[node] = TyNone()

        assert node in self.nodetype.keys(), type(node).__name__
        return self.nodetype[node]


    def infer_FunctionDef(self, node):
        # FunctionDef(identifier name, arguments args, stmt* body,
        #             expr* decorator_list, expr? returns)
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
            if ty_val.is_mutable() and \
                    isinstance(node.value, (gast.Name, gast.Attribute)):
                # XXX: alias
                self.tyenv[target.id] = ty_val
                self.nodetype[target] = ty_val
                return

            # XXX: Changing following 2 lines into
            #   self.tyenv[target.id] = self.nodetype[target] = copy_ty(ty_val)
            # will allow self.nodetype[target] to change afterwards, which will
            # be more suitable for elichika but contradict with python semantics.
            self.tyenv[target.id] = copy_ty(ty_val)
            self.nodetype[target] = copy_ty(ty_val)
            return

        if isinstance(target, gast.Attribute):
            self.infer_expr(target.value)
            ty_obj = self.nodetype[target.value]
            assert isinstance(ty_obj, TyUserDefinedClass)
            self.attribute_tyenv[(ty_obj.instance, target.attr)] = ty_val
            self.nodetype[target] = ty_val
            return

        if isinstance(target, (gast.Tuple, gast.List)):
            if isinstance(target, gast.Tuple):
                ty_target = TyTuple([self.generate_fresh_TyVar(e) for e in target.elts])
            else:
                ty_target = TyList([self.generate_fresh_TyVar(e) for e in target.elts])
            self.nodetype[target] = ty_target
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
            unify(ty_i, TyTensor(ty_iteration.kind, ty_iteration.dtype,
                ty_iteration.shape[1:]))
        else:
            unify(ty_iteration, TySequence(ty_i, None))

        for _ in range(2):
            tc = copy_InferenceEngine(self)
            self.infer_block(tc, node.body)

        utils.add_dict(self.nodetype, tc.nodetype)
        utils.add_dict(self.subroutine_node, tc.subroutine_node)

    def infer_If(self, node):
        # If(expr test, stmt* body, stmt* orelse)
        # XXX: type of node.test can be anything
        self.infer_expr(node.test)
        x = lazy_initializer(node)

        if node.orelse == []:
            tc = copy_InferenceEngine(self)
            ty_ret = self.infer_block(tc, node.body)
            utils.add_dict(self.nodetype, tc.nodetype)
            utils.add_dict(self.subroutine_node, tc.subroutine_node)
        else:
            tc1 = copy_InferenceEngine(self)
            tc2 = copy_InferenceEngine(self)
            ty_ret = self.infer_2blocks(tc1, tc2, node.body, node.orelse)
            utils.add_dict(self.nodetype, tc1.nodetype)
            utils.add_dict(self.nodetype, tc2.nodetype)
            utils.add_dict(self.subroutine_node, tc1.subroutine_node)
            utils.add_dict(self.subroutine_node, tc2.subroutine_node)

        if isinstance(x, gast.Name):
            self.tyenv[x.id].is_optional = False
        elif isinstance(x, gast.Attribute):
            obj = self.infer_expr(x.value).instance
            self.attribute_tyenv[(obj, x.attr)].is_optional = False

        return ty_ret


    # ================================= expr ===================================
    def infer_expr(self, node, is_callee=False):
        if node in self.nodetype.keys():
            return self.nodetype[node]

        if self.is_debug:
            pass
            # debug(gast.dump(node))
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
            self.infer_expr(node.left)
            for comparator in node.comparators:
                self.infer_expr(comparator)
            self.nodetype[node] = TyBool()
        elif isinstance(node, gast.Call):
            self.nodetype[node] = self.infer_Call(node)
        elif isinstance(node, gast.Constant):
            # Constant(constant value)
            self.nodetype[node] = type_of_value(node.value)
        elif isinstance(node, gast.Attribute):
            self.nodetype[node] = self.infer_Attribute(node, is_callee)
        elif isinstance(node, gast.Subscript):
            self.nodetype[node] = self.infer_Subscript(node)
        elif isinstance(node, gast.Name):
            self.nodetype[node] = self.infer_Name(node, is_callee)
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
        if self.is_debug:
            self.dump_one_node(node)
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
        gen = node.generators[0]
        # TODO: handle cases where len(gen.ifs) > 0
        assert len(gen.ifs) == 0

        tc = copy_InferenceEngine(self)
        ty_iteration = tc.infer_expr(gen.iter)
        ty_i = tc.generate_fresh_TyVar(gen.target)
        if isinstance(ty_iteration, TyTensor):
            ty_i_ = TyTensor(ty_iteration.kind, ty_iteration.dtype,
                    ty_iteration.shape[1:])
            if ty_iteration.shape is not None:
                ty_i_.shape = ty_iteration.shape[1:]
            unify(ty_i, ty_i_)
        else:
            unify(TySequence(ty_i, None), ty_iteration)
        tc.infer_expr(node.elt)

        utils.add_dict(self.nodetype, tc.nodetype)
        utils.add_dict(self.subroutine_node, tc.subroutine_node)

        self.nodetype[node] = TyList(tc.nodetype[node.elt])
        return self.nodetype[node]


    def infer_Call(self, node):
        # Call(expr func, expr* args, keyword* keywords)

        # XXX: no need to deref() argument type later on
        ty_args = [self.infer_expr(arg).deref() for arg in node.args]
        ty_kwargs = {kwarg.arg : self.infer_expr(kwarg.value) \
                for kwarg in node.keywords}
        ty_ret = TyVar()

        try:
            ty_fun = self.infer_expr(node.func, is_callee=True)
            unify(ty_fun, TyArrow(ty_args, ty_ret))
        except self.ArgumentRequired as e:
            # Attribute
            if isinstance(e.func, tuple):
                (func, ty_obj) = e.func
                e.func = func
                ty_args_ = [ty_obj] + ty_args
            else:
                ty_args_ = ty_args

            if e.func in func_to_ignore:
                return TyNone()

            ty_ret = self.infer_function_instance(
                    node, e.func, ty_args_, ty_kwargs)

        self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
        return ty_ret.deref()


    def infer_Attribute(self, node, is_callee):
        # Attribute(expr value, identifier attr, expr_context ctx)

        if isinstance(node.value, gast.Name) and \
                hasattr(self.module, node.value.id):
            # function of imported libraries (eg. np, chainer, F, L)
            module = getattr(self.module, node.value.id)
            attr = getattr(module, node.attr)
            if is_callee:
                raise self.ArgumentRequired(func=attr)
            return type_of_value(attr)

        ty_obj = self.infer_expr(node.value).deref()

        if isinstance(ty_obj, TySequence) and ty_obj.is_list():
            ty_obj.coerce_to_variable_len()
            return list_attr_ty[node.attr](ty_obj)

        if isinstance(ty_obj, TyTensor):
            # TODO: compare by numpy objects, not names
            if node.attr == 'shape':
                if ty_obj.shape is None:
                    return TyTuple(TyInt())
                return type_of_value(ty_obj.shape)
            if node.attr == 'size':
                return TyInt()
            if ty_obj.is_ndarray() and is_callee:
                func = getattr(np.ndarray, node.attr)
                raise self.ArgumentRequired((func, ty_obj))
            if ty_obj.is_torch_tensor() and is_callee:
                func = getattr(torch.Tensor, node.attr)
                raise self.ArgumentRequired((func, ty_obj))
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

            if is_callee:
                raise self.ArgumentRequired(func=x)

            return ty_node

        if isinstance(ty_obj, TyNone):
            return TyVar()


    def infer_Subscript(self, node):
        # Subscript(expr value, slice slice, expr_context ctx)
        ty_obj = self.infer_expr(node.value)

        if isinstance(ty_obj, TySequence):
            self.infer_slice(node.slice)

            if ty_obj.is_fixed_len and \
                    isinstance(node.slice, gast.Index):
                t = self.infer_expr(node.slice.value)
                if isinstance(t, TyNum) and t.value is not None:
                    return ty_obj.get_tys()[t.value]

            if ty_obj.is_fixed_len and \
                    isinstance(node.slice, gast.Slice) and \
                    self.is_const_slice(node.slice):
                slice_ = self.extract_slice(node.slice)
                if ty_obj.is_list():
                    return TyList(ty_obj.get_tys()[slice_])
                return TyTuple(ty_obj.get_tys()[slice_])

            ty_obj.coerce_to_variable_len()
            if isinstance(node.slice, gast.Index):
                return ty_obj.get_ty()
            if isinstance(node.slice, gast.Slice):
                return ty_obj
            assert False, "ExtSlice for lists/tuples is not supported"

        if isinstance(ty_obj, TyDict):
            self.infer_slice(node.slice, ty_obj.keyty)
            assert isinstance(node.slice, gast.Index)
            return ty_obj.valty

        if isinstance(ty_obj, TyTensor):
            self.infer_slice(node.slice)
            ret_shape = self.infer_Subscript_shape(ty_obj.shape, node.slice)
            return TyTensor(ty_obj.kind, ty_obj.dtype, ret_shape)


    def infer_Subscript_shape(self, shape, node_slice):
        if isinstance(node_slice, gast.Index):
            return shape[1:]
        if isinstance(node_slice, gast.Slice):
            if not self.is_const_slice(node_slice):
                return (None,) + shape[1:]
            if shape[0].value is None and (node_slice.upper is None or
                    extract_value_from_ty(self.nodetype[node_slice.upper]) < 0):
                return (None,) + shape[1:]
            slice_ = self.extract_slice(node_slice)
            shape_0 = ShapeElem(len(((0,) * shape[0].value)[slice_]))
            return (shape_0,) + shape[1:]
        if isinstance(node_slice, gast.ExtSlice):
            ret_shape = ()
            for i in range(len(node_slice.dims)):
                ret_shape += self.infer_Subscript_shape(shape[i:i+1],
                        node_slice.dims[i])
            ret_shape += shape[len(node_slice.dims):]
            return ret_shape


    def infer_Name(self, node, is_callee):
        # Name(identifier id, expr_context ctx, expr? annotation)
        if node.id in self.tyenv.keys():
            ty = self.tyenv[node.id]
            if is_callee and isinstance(ty, TyUserDefinedClass) and \
                    callable(ty.instance):
                raise self.ArgumentRequired(func=ty.instance)
            return self.tyenv[node.id]
        if node.id in __builtins__.keys():
            value = __builtins__[node.id]
            if callable(value) and is_callee:
                raise self.ArgumentRequired(func=value)
            return type_of_value(value)
        if hasattr(self.module, node.id):
            x = getattr(self.module, node.id)
            if is_callee:
                raise self.ArgumentRequired(func=x)
            return type_of_value(x)

        # XXX: print comes here
        ty_var = TyVar()
        self.tyenv[node.id] = ty_var
        return ty_var


    # ================================= slice ==================================
    def infer_slice(self, node, ty_key_expected=TyInt()):
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

        if isinstance(node, gast.ExtSlice):
            # ExtSlice(slice* dims)
            for s in node.dims:
                self.infer_slice(s, ty_key_expected)


    def is_const_slice(self, node_slice):
        is_constnum = lambda t: isinstance(t, TyNum) and t.value is not None

        if node_slice.lower and not is_constnum(self.infer_expr(node_slice.lower)):
            return False
        if node_slice.upper and not is_constnum(self.infer_expr(node_slice.upper)):
            return False
        if node_slice.step and not is_constnum(self.infer_expr(node_slice.step)):
            return False
        return True

    def extract_slice(self, node_slice) -> slice:
        lower, upper, step = None, None, None
        if node_slice.lower:
            lower = self.infer_expr(node_slice.lower).value
        if node_slice.upper:
            upper = self.infer_expr(node_slice.upper).value
        if node_slice.step:
            step = self.infer_expr(node_slice.step).value
        return slice(lower, upper, step)


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
        code = clip_head(inspect.getsource(func))
    else:
        module = None
        code = open(sys.argv[1]).read()
    orig_ast = gast.ast_to_gast(ast.parse(code))
    dump_ast(orig_ast, 'original')

    tc = InferenceEngine(is_debug=True, module=module)
    try:
        nodetype = tc.infer(orig_ast)
    except UnifyError as e:
        print(traceback.format_exc(), end="")
