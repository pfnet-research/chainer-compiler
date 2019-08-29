import ast
import inspect
import gast
import os
import traceback
import types
from copy import deepcopy
from pprint import pprint

from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser.types import *

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

def debug(sth):
    frame = inspect.currentframe().f_back
    print("[{} {}] {}".format(frame.f_code.co_name, frame.f_lineno, sth))


def defined_with___call__(func):
    return not isinstance(func, (types.FunctionType, types.LambdaType,
        types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType))


def callable_(x):
    # TODO(momohatt): この分類どうしよう
    if isinstance(x, L.Linear):
        return False
    return callable(x)



# ==============================================================================

builtins_name = ['float', 'range', 'abs']

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
        }


ext_func_ty = {
        np.array :
            (lambda x: TyArrow([x],
                TyNdarray(np.dtype(pytype_of_type(x.get_ty()))))) \
                        (TySequence(TyBool())),
        np.ones :
            TyArrow([TyUnion(TyBool(), TySequence(TyIntOnly()))],
                TyNdarray(np.dtype('float64'))),
        np.zeros :
            TyArrow([TyUnion(TyBool(), TySequence(TyIntOnly()))],
                TyNdarray(np.dtype('float64'))),
        F.relu :
            (lambda x: TyArrow([x], TyChainerVariable(x.dtype))) \
                    (TyTensor()),
        F.softmax_cross_entropy :
            TyArrow([TyTensor(), TyTensor()],
                TyChainerVariable(np.dtype('float32'))),
        }


list_attr_ty = {
        'append'  : (lambda x: TyArrow([x, x.get_ty()], TyNone()))(TyList(TyVar())),
        'reverse' : TyArrow([TyList(TyVar())], TyNone()),
        }


def ty_NumOp(tyl, tyr):
    if isinstance(tyl, TyNum) and isinstance(tyr, TyNum):
        return TyNum(max(tyl.ty_level_min, tyr.ty_level_min), 2)
    assert False

def ty_Add(tyl, tyr):
    if isinstance(tyl, TyNum) and isinstance(tyr, TyNum):
        return TyNum(max(tyl.ty_level_min, tyr.ty_level_min), 2)
    if isinstance(tyl, TyString) and isinstance(tyr, TyString):
        return TyString()
    if isinstance(tyl, TySequence) and isinstance(tyr, TySequence) and \
            tyl.seq_kind == tyr.seq_kind:
        ty = TyVar()
        unify(tyl, TyList(ty))
        unify(tyr, TyList(ty))
        tyl.coerce_to_variable_len(ty)
        tyr.coerce_to_variable_len(ty)
        return TySequence(ty, tyl.seq_kind)
    assert False

def ty_Div(tyl, tyr):
    if isinstance(tyl, TyNum) and isinstance(tyr, TyNum):
        return TyFloat()
    assert False


# binop も次のようにしたいが、dictの初期化時に
# max(x.ty_level_min, y.ty_level_min)とかの値が決まってしまうので難しい...
# binop_ty = {
#         gast.Add : TyUnion(
#             (lambda x, y: TyArrow([x, y],
#                 TyNum(max(x.ty_level_min, y.ty_level_min), 2))) \
#                         (TyBool(), TyBool()),
#             TyArrow([TyString(), TyString()], TyString()),
#             (lambda x: TyArrow([TyList(x), TyList(x)], TyList(x)))(TyVar()),
#             (lambda x: TyArrow([TyTuple(x), TyTuple(x)], TyTuple(x)))(TyVar()),
#             ),
#         }


primitive_op_ty = {
        gast.Add : ty_Add,
        gast.Sub : ty_NumOp,
        gast.Mult : ty_NumOp,
        gast.Div : ty_Div,
        gast.FloorDiv : ty_NumOp,
        }


# ==============================================================================

class TypeChecker():
    class ArgumentRequired(Exception):
        def __init__(self, func):
            self.func = func

    def __init__(self, tyenv=None, is_debug=False, module=None):
        if tyenv is None:
            self.tyenv = {}  # string -> TyObj (internal type env)
        else:
            self.tyenv = deepcopy(tyenv)
        # type environments
        self.nodetype = {}  # Node -> TyObj (for elichika to use)
        self.is_debug = is_debug
        self.module = module
        self.subroutine_node = {}  # Node (Call) -> Node (FunctionDef)


    def dump_tyenv(self):
        if not self.is_debug:
            return
        for name, ty in self.tyenv.items():
            print(name + " : \x1b[35m" + str(ty) + "\x1b[39m")
        print()


    def dump_nodetype(self):
        if not self.is_debug:
            return
        for node, ty in self.nodetype.items():
            print(gast.dump(node) + " : \x1b[36m" + str(ty) + "\x1b[39m")
        print()


    def infer(self, node: 'gast.Node'):
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


    def infer_function_vargs(self, node, args):
        # args: argument value
        ty_args = [type_of_value(arg) for arg in args]
        return self.infer_function(node, ty_args)


    def infer_function(self, node: 'gast.Node', ty_args):
        assert isinstance(node, gast.FunctionDef)
        assert len(ty_args) == len(node.args.args), \
            "Wrong number of arguments"

        for arg_node, ty in zip(node.args.args, ty_args):
            self.tyenv[arg_node.id] = ty

        self.infer_stmt(node)

        if self.is_debug:
            print('=== Type Environment ===')
            self.dump_nodetype()

        pprint(self.nodetype)
        pprint(self.subroutine_node)

        return self.nodetype


    # ================================ mod =====================================
    def infer_mod(self, node: 'gast.Node'):
        if isinstance(node, gast.Module):
            self.infer_stmt(node.body[0])
            return

        assert False


    # ================================ stmt ====================================
    def infer_stmt(self, node: 'gast.Node') -> 'TyObj':
        if self.is_debug:
            debug(gast.dump(node))
            self.dump_tyenv()

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

            if isinstance(target, gast.Name):
                ty_val = self.infer_expr(node.value)
                if isinstance(node.value, gast.Name):
                    self.tyenv[target.id] = ty_val
                    self.nodetype[target] = ty_val
                else:
                    self.tyenv[target.id] = deepcopy(ty_val)
                    self.nodetype[target] = deepcopy(ty_val)
            elif isinstance(target, gast.Attribute):
                pass
            elif type(target) in [gast.Tuple, gast.List]:
                ty_target = self.infer_expr(target)
                ty_val = self.infer_expr(node.value)
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
            if self.tyenv[node.target.id].is_mutable():
                binop = gast.BinOp(node.target, node.op, node.value)
                ty_val = self.infer_expr(binop)
                del self.nodetype[binop]
            else:
                self.tyenv[node.target.id] = deepcopy(self.tyenv[node.target.id])
                binop = gast.BinOp(node.target, node.op, node.value)
                ty_val = self.infer_expr(binop)
                del self.nodetype[binop]
            self.tyenv[node.target.id] = ty_val
            self.nodetype[node.target] = ty_val
            self.nodetype[node] = TyNone()
            return self.nodetype[node]


        if isinstance(node, gast.For):
            # For(expr target, expr iter, stmt* body, stmt* orelse)
            assert type(node.target) in [gast.Name, gast.Tuple]

            ty_iteration = self.infer_expr(node.iter)
            ty_i = self.infer_expr(node.target)
            unify(ty_iteration, TyList(ty_i))
            ty_iteration.coerce_to_variable_len(ty_i)

            for stmt in node.body:
                self.infer_stmt(stmt)

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
                tc = TypeChecker(self.tyenv, is_debug=self.is_debug)
                for stmt in node.body:
                    tc.infer_stmt(stmt)

                # 1. unify the intersection of 2 tyenvs
                for name, ty in tc.tyenv.items():
                    if name in self.tyenv.keys():
                        unify(ty, self.tyenv[name])

                # 2. update local tyenv
                for name, ty in tc.tyenv.items():
                    if name in self.tyenv.keys():
                        self.tyenv[name] = ty

                # 3. merge nodetype from 2 TypeCheckers
                for node_, ty in tc.nodetype.items():
                    self.nodetype[node_] = ty
            else:
                tc1 = TypeChecker(self.tyenv, is_debug=self.is_debug)
                tc2 = TypeChecker(self.tyenv, is_debug=self.is_debug)
                for stmt in node.body:
                    tc1.infer_stmt(stmt)
                for stmt in node.orelse:
                    tc2.infer_stmt(stmt)

                # 1. unify the intersection of 2 tyenvs
                for name, ty in tc1.tyenv.items():
                    if name in tc2.tyenv.keys():
                        unify(ty, tc2.tyenv[name])

                # 2. update local tyenv
                for name, ty in tc1.tyenv.items():
                    if name in tc2.tyenv.keys():
                        self.tyenv[name] = ty

                # 3. merge nodetype from 2 TypeCheckers
                for node_, ty in tc1.nodetype.items():
                    self.nodetype[node_] = ty
                for node_, ty in tc2.nodetype.items():
                    self.nodetype[node_] = ty

            self.nodetype[node] = TyNone()
            return self.nodetype[node]


        if isinstance(node, gast.Expr):
            # Expr(expr value)
            self.nodetype[node] = self.infer_expr(node.value)
            return self.nodetype[node]


        if isinstance(node, gast.Pass):
            self.nodetype[node] = TyNone()
            return self.nodetype[node]

        assert False


    # ================================= expr ===================================
    def infer_expr(self, node: 'gast.Node') -> 'TyObj':
        if self.is_debug:
            debug(gast.dump(node))
            self.dump_tyenv()

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

            ty_ret = primitive_op_ty[type(node.op)](tyl, tyr)
            self.nodetype[node.op] = TyArrow([tyl, tyr], ty_ret)
            self.nodetype[node] = ty_ret
            return self.nodetype[node]

        if isinstance(node, gast.UnaryOp):
            # UnaryOp(unaryop op, expr operand)
            # TODO
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
            return self.nodetype[node]


        if isinstance(node, gast.Call):
            # Call(expr func, expr* args, keyword* keywords)
            ty_args = [self.infer_expr(arg) for arg in node.args]
            ty_ret = TyVar()

            try:
                ty_fun = self.infer_expr(node.func)
            except self.ArgumentRequired as e:
                # cases where argument info is necessary to type function
                if defined_with___call__(e.func):
                    code = utils.clip_head(inspect.getsource(e.func.__call__))
                    ty_self = self.nodetype[node.func]
                    ty_args = [ty_self] + ty_args

                else:
                    code = utils.clip_head(inspect.getsource(e.func))

                    if isinstance(node.func, gast.Attribute):
                        ty_self = self.nodetype[node.func.value]
                        ty_args = [ty_self] + ty_args

                # FunctionDef of called subroutine
                func_node = gast.ast_to_gast(ast.parse(code)).body[0]
                self.subroutine_node[node] = func_node
                tc = TypeChecker(module=self.module)
                tc.infer_function(func_node, ty_args)

                # copy nodetype and subroutine_node from subroutine
                for k, v in tc.nodetype.items():
                    self.nodetype[k] = v

                for k, v in tc.subroutine_node.items():
                    self.subroutine_node[k] = v

                ty_fun = tc.nodetype[func_node]
                self.nodetype[node.func] = ty_fun


            unify(ty_fun, TyArrow(ty_args, ty_ret))
            self.nodetype[node] = ty_ret.deref()

            return self.nodetype[node]


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
            # Attribute(expr value, identifier attr, expr_context ctx)

            if isinstance(node.value, gast.Name) and \
                    hasattr(self.module, node.value.id):
                module = getattr(self.module, node.value.id)
                self.nodetype[node] = deepcopy(ext_func_ty[getattr(module, node.attr)])
                return self.nodetype[node]

            ty_obj = self.infer_expr(node.value)

            if isinstance(ty_obj, TySequence) and ty_obj.is_list():
                ty_fun = deepcopy(list_attr_ty[node.attr])
                self.nodetype[node.attr] = ty_fun
                self.nodetype[node] = TyArrow(ty_fun.argty[1:], ty_fun.retty)
                unify(ty_fun.argty[0], ty_obj)
                return self.nodetype[node]

            if isinstance(ty_obj, TyUserDefinedClass):
                # x: value of existing instance
                x = getattr(ty_obj.instance, node.attr)
                if callable_(x):
                    if defined_with___call__(x):
                        self.nodetype[node] = type_of_value(x)
                    raise self.ArgumentRequired(x)
                self.nodetype[node] = type_of_value(x)
                return self.nodetype[node]

            assert False


        if isinstance(node, gast.Subscript):
            # Subscript(expr value, slice slice, expr_context ctx)
            ty_obj = self.infer_expr(node.value)

            if isinstance(ty_obj, TySequence):
                self.infer_slice(node.slice, TyInt())
                if ty_obj.is_fixed_len:
                    if isinstance(node.slice, gast.Index) and \
                            isinstance(node.slice.value, gast.Num):
                        # TODO(momohatt): handle cases where index is
                        # more complex but still a constant
                        self.nodetype[node] = ty_obj.get_tys()[node.slice.value.n]
                    else:
                        ty_obj.coerce_to_variable_len()
                        if isinstance(node.slice, gast.Index):
                            self.nodetype[node] = ty_obj.get_ty()
                        elif isinstance(node.slice, gast.Slice):
                            self.nodetype[node] = ty_obj
                        else:
                            assert False

                else:
                    if isinstance(node.slice, gast.Index):
                        self.nodetype[node] = ty_obj.get_ty()
                    elif isinstance(node.slice, gast.Slice):
                        self.nodetype[node] = ty_obj
                    else:
                        assert False

            elif isinstance(ty_obj, TyDict):
                self.infer_slice(node.slice, ty_obj.keyty)
                assert isinstance(node.slice, gast.Index)
                self.nodetype[node] = ty_obj.valty

            elif isinstance(ty_obj, TyNdarray):
                self.infer_slice(node.slice, TyInt())
                if isinstance(node.slice, gast.Index):
                    self.nodetype[node] = ty_obj.ty
                elif isinstance(node.slice, gast.Slice):
                    self.nodetype[node] = ty_obj
                else:
                    assert False

            else:
                assert False
            return self.nodetype[node]


        if isinstance(node, gast.Name):
            # Name(identifier id, expr_context ctx, expr? annotation)
            if node.id in self.tyenv.keys():
                self.nodetype[node] = self.tyenv[node.id]
            elif node.id in builtins_name:
                self.nodetype[node] = deepcopy(builtins_ty[eval(node.id)])
            elif hasattr(self.module, node.id):
                x = getattr(self.module, node.id)
                if callable_(x):
                    raise self.ArgumentRequired(x)
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

        assert False


    def infer_slice(self, node: 'gast.Node', ty_key_expected) -> 'NoneType':
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

    code = open(sys.argv[1]).read()
    orig_ast = gast.ast_to_gast(ast.parse(code))
    dump_ast(orig_ast, 'original')

    is_debug_global = True
    tc = TypeChecker(is_debug=True)
    try:
        nodetype = tc.infer(orig_ast)
    except UnifyError as e:
        print(traceback.format_exc(), end="")
