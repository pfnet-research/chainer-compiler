import ast
import inspect
import gast
import os
import traceback
from copy import deepcopy

from chainer_compiler.elichika.parser import utils
import chainer_compiler.elichika.parser.types as T

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

def debug(sth):
    frame = inspect.currentframe().f_back
    print("[{} {}] {}".format(frame.f_code.co_name, frame.f_lineno, sth))

# ==============================================================================

builtins_name = ['float', 'range', 'abs']

builtins_ty = {
        float : T.TyArrow([T.TyBool()], T.TyFloat()),
        # int -> int \/ int -> int -> int \/ int -> int -> int -> int
        range : T.TyUnion(
            T.TyArrow([T.TyIntOnly()], T.TyList(T.TyIntOnly())),
            T.TyArrow([T.TyIntOnly(), T.TyIntOnly()], T.TyList(T.TyIntOnly())),
            T.TyArrow([T.TyIntOnly(), T.TyIntOnly(), T.TyIntOnly()], T.TyList(T.TyIntOnly())),
            ),
        abs : T.TyUnion(
            T.TyArrow([T.TyIntOnly()], T.TyIntOnly()),
            T.TyArrow([T.TyFloat()], T.TyFloat()),
            ),
        }


def ty_NumpyArray(ty_args):
    assert len(ty_args) == 1
    ty = ty_args[0]
    assert isinstance(ty, T.TySequence)

    ty.coerce_to_variable_len()
    return T.TyNdarray(np.dtype(pytype_of_type(ty.get_ty())))


def ty_NumpyOnes(ty_args):
    assert len(ty_args) == 1
    ty = ty_args[0]

    if isinstance(ty, T.TyNum):
        return T.TyNdarray(np.dtype('float64'))

    if isinstance(ty, T.TySequence):
        assert ty.is_fixed_len
        return T.TyNdarray(np.dtype('float64'))

    assert False


def ty_ChainerReLU(ty_args):
    assert len(ty_args) == 1
    ty = ty_args[0].deref()

    if isinstance(ty, T.TyTensor):
        return T.TyChainerVariable(ty.dtype)

    assert False


def ty_ChainerSoftmaxCrossEntropy(ty_args):
    assert len(ty_args) == 2
    ty0 = ty_args[0].deref()
    ty1 = ty_args[1].deref()

    if isinstance(ty0, T.TyTensor) and isinstance(ty1, T.TyTensor):
        return T.TyChainerVariable(np.dtype('float32'))


ext_func_ty = {
        np.array : ty_NumpyArray,
        np.ones : ty_NumpyOnes,
        np.zeros : ty_NumpyOnes,
        F.relu : ty_ChainerReLU,
        F.softmax_cross_entropy : ty_ChainerSoftmaxCrossEntropy,
        }


list_attr_ty = {
        'append'  : lambda ty_obj: T.TyArrow([ty_obj.get_ty()], T.TyNone()),
        'reverse' : lambda ty_obj: T.TyArrow([ty_obj], T.TyNone()),
        }


def ty_NumOp(tyl, tyr):
    if isinstance(tyl, T.TyNum) and isinstance(tyr, T.TyNum):
        return T.TyNum(max(tyl.ty_level_min, tyr.ty_level_min), 2)
    assert False

def ty_Add(tyl, tyr):
    # match tyl, tyr with
    # | T.TyNum(n, _), T.TyNum(m, _) -> T.TyNum(max(n, m), 2)
    # | T.TyString(), T.TyString() -> T.TyString
    # | T.TyList(), T.TyList() -> T.TyList
    if isinstance(tyl, T.TyNum) and isinstance(tyr, T.TyNum):
        return T.TyNum(max(tyl.ty_level_min, tyr.ty_level_min), 2)
    if isinstance(tyl, T.TyString) and isinstance(tyr, T.TyString):
        return T.TyString()
    if isinstance(tyl, T.TySequence) and isinstance(tyr, T.TySequence) and \
            tyl.seq_kind == tyr.seq_kind:
        ty = T.TyVar()
        T.unify(tyl, T.TyList(ty))
        T.unify(tyr, T.TyList(ty))
        tyl.coerce_to_variable_len(ty)
        tyr.coerce_to_variable_len(ty)
        return T.TySequence(tyl.seq_kind, ty)
    assert False

def ty_Div(tyl, tyr):
    if isinstance(tyl, T.TyNum) and isinstance(tyr, T.TyNum):
        return T.TyFloat()
    assert False


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
        def __init__(self, ty_obj, func):
            self.ty_obj = ty_obj
            self.func = func

    class InlineRequired(Exception):
        def __init__(self, term):
            assert isinstance(term, gast.Call)
            self.term = term
            self.func = term.func
            self.args = term.args

    def __init__(self, tyenv=None, is_debug=False, module=None):
        if tyenv is None:
            self.tyenv = {}  # string -> TyObj (internal type env)
        else:
            self.tyenv = deepcopy(tyenv)
        # type environments
        self.nodetype = {}  # Node -> TyObj (for elichika to use)
        self.is_debug = is_debug
        self.module = module
        self.args = {}  # argument values


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


    def infer(self, node: 'ast.Node') -> 'TyObj':
        """
        Adds local type information to self.tyenv while traversing the AST
        returns: type
        """
        self.infer_mod(node)

        if self.is_debug:
            print('=== Type Environment ===')
            self.dump_nodetype()

        return self.nodetype


    def infer_function_vargs(self, node, args) -> 'TyObj':
        # args: argument value
        ty_args = [T.type_of_value(arg) for arg in args]

        for arg_node, arg_value, ty in zip(node.args.args, args, ty_args):
            self.tyenv[arg_node.id] = ty
            self.args[arg_node.id] = arg_value

        return self.infer_function(node, ty_args)


    def infer_function(self, node: 'ast.Node', ty_args) -> 'TyObj':
        assert isinstance(node, gast.FunctionDef)
        assert len(ty_args) == len(node.args.args)

        # examine argument type separately from parent typechecker
        tc = TypeChecker()

        self.infer_stmt(node)

        if self.is_debug:
            print('=== Type Environment ===')
            self.dump_nodetype()

        return self.nodetype


    # ================================ mod =====================================
    def infer_mod(self, node: 'ast.Node'):
        if isinstance(node, gast.Module):
            self.infer_stmt(node.body[0])
        else:
            assert False


    # ================================ stmt ====================================
    def infer_stmt(self, node: 'ast.Node') -> 'TyObj':
        if self.is_debug:
            debug(gast.dump(node))
            self.dump_tyenv()

        if isinstance(node, gast.FunctionDef):
            # FunctionDef(identifier name, arguments args, stmt* body,
            # expr* decorator_list, expr? returns)

            ty_args = [self.tyenv[arg.id] for arg in node.args.args[1:]]

            for stmt in node.body:
                try:
                    ty = self.infer_stmt(stmt)
                except self.InlineRequired as e:
                    print(e.term)
                    print(e.func)
                    print(e.args)
                    pass

            # TODO(momohatt): type of function definition?
            self.nodetype[node] = T.TyArrow(ty_args, ty)


        elif isinstance(node, gast.Return):
            # Return(expr? value)
            self.nodetype[node] = self.infer_expr(node.value)


        elif isinstance(node, gast.Assign):
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
                T.unify(ty_target, ty_val)
                for (var, ty) in zip(target.elts, ty_val.get_tys()):
                    self.tyenv[var.id] = ty
                    self.nodetype[var] = ty
            else:
                assert False

            self.nodetype[node] = T.TyNone()


        elif isinstance(node, gast.AugAssign):
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
            self.nodetype[node] = T.TyNone()


        elif isinstance(node, gast.For):
            # For(expr target, expr iter, stmt* body, stmt* orelse)
            assert type(node.target) in [gast.Name, gast.Tuple]

            ty_iteration = self.infer_expr(node.iter)
            ty_i = self.infer_expr(node.target)
            T.unify(ty_iteration, T.TyList(ty_i))
            ty_iteration.coerce_to_variable_len(ty_i)

            for stmt in node.body:
                self.infer_stmt(stmt)

            self.nodetype[node] = T.TyNone()


        elif isinstance(node, gast.While):
            # While(expr test, stmt* body, stmt* orelse)
            pass


        elif isinstance(node, gast.If):
            # If(expr test, stmt* body, stmt* orelse)
            ty_test = self.infer_expr(node.test)
            # TODO(momohatt): determine what type should ty_test be

            if node.orelse == []:
                tc = TypeChecker(self.tyenv)
                for stmt in node.body:
                    tc.infer_stmt(stmt)

                # 1. unify the intersection of 2 tyenvs
                for name, ty in tc.tyenv.items():
                    if name in self.tyenv.keys():
                        T.unify(ty, self.tyenv[name])

                # 2. update local tyenv
                for name, ty in tc.tyenv.items():
                    if name in self.tyenv.keys():
                        self.tyenv[name] = ty

                # 3. merge nodetype from 2 TypeCheckers
                for node_, ty in tc.nodetype.items():
                    self.nodetype[node_] = ty
            else:
                tc1 = TypeChecker(self.tyenv)
                tc2 = TypeChecker(self.tyenv)
                for stmt in node.body:
                    tc1.infer_stmt(stmt)
                for stmt in node.orelse:
                    tc2.infer_stmt(stmt)

                # 1. unify the intersection of 2 tyenvs
                for name, ty in tc1.tyenv.items():
                    if name not in tc2.tyenv.keys():
                        continue
                    # untypeable If-stmts will raise error here
                    T.unify(ty, tc2.tyenv[name])

                # 2. update local tyenv
                for name, ty in tc1.tyenv.items():
                    if name not in tc2.tyenv.keys():
                        continue
                    self.tyenv[name] = ty

                # 3. merge nodetype from 2 TypeCheckers
                for node_, ty in tc1.nodetype.items():
                    self.nodetype[node_] = ty
                for node_, ty in tc2.nodetype.items():
                    self.nodetype[node_] = ty

            self.nodetype[node] = T.TyNone()


        elif isinstance(node, gast.Expr):
            # Expr(expr value)
            self.nodetype[node] = self.infer_expr(node.value)


        elif isinstance(node, gast.Pass):
            self.nodetype[node] = T.TyNone()


        return self.nodetype[node]


    # ================================= expr ===================================
    def infer_expr(self, node: 'ast.Node') -> 'TyObj':
        if self.is_debug:
            debug(gast.dump(node))
            self.dump_tyenv()

        if isinstance(node, gast.BoolOp):
            # BoolOp(boolop op, expr* values)
            ty_vals = [self.infer_expr(val) for val in node.values]
            for ty in ty_vals:
                T.unify(ty, T.TyBool())
            self.nodetype[node.op] = T.TyArrow([T.TyBool(), T.TyBool()], T.TyBool())
            self.nodetype[node] = T.TyBool()


        elif isinstance(node, gast.BinOp):
            # BinOp(expr left, operator op, expr right)
            tyl = self.infer_expr(node.left).deref()
            tyr = self.infer_expr(node.right).deref()

            ty_ret = primitive_op_ty[type(node.op)](tyl, tyr)
            self.nodetype[node.op] = T.TyArrow([tyl, tyr], ty_ret)
            self.nodetype[node] = ty_ret


        elif isinstance(node, gast.UnaryOp):
            # UnaryOp(unaryop op, expr operand)
            pass


        elif isinstance(node, gast.Dict):
            # Dict(expr* keys, expr* values)
            if node.keys == []:
                self.nodetype[node] = T.TyDict(T.TyVar(), T.TyVar())
            else:
                ty_keys = [self.infer_expr(key) for key in node.keys]
                ty_vals = [self.infer_expr(val) for val in node.values]
                assert T.all_same_ty(ty_keys)
                assert T.all_same_ty(ty_vals)
                self.nodetype[node] = T.TyDict(ty_keys[0], ty_vals[0])


        elif isinstance(node, gast.Compare):
            # Compare(expr left, cmpop* ops, expr* comparators)
            pass


        elif isinstance(node, gast.Call):
            # Call(expr func, expr* args, keyword* keywords)
            ty_args = [self.infer_expr(arg) for arg in node.args]
            ty_ret = T.TyVar()

            if isinstance(node.func, gast.Attribute) and \
                    isinstance(node.func.value, gast.Name) and \
                    hasattr(self.module, node.func.value.id):
                module = getattr(self.module, node.func.value.id)
                ty_ret = ext_func_ty[getattr(module, node.func.attr)](ty_args)
                ty_ret = ty_ret.deref()
                self.nodetype[node.func] = T.TyArrow(ty_args, ty_ret)
                self.nodetype[node] = ty_ret

            else:
                try:
                    ty_fun = self.infer_expr(node.func)
                    T.unify(ty_fun, T.TyArrow(ty_args, ty_ret))
                    self.nodetype[node] = ty_ret.deref()
                except self.ArgumentRequired as e:
                    # cases where argument info is necessary to type function
                    raise self.InlineRequired(node)
                #     code = utils.clip_head(inspect.getsource(x))
                #     func_node = gast.ast_to_gast(ast.parse(code))
                #     ty_fun = self.infer_function(func_node.body[0], [e.ty_obj] + ty_args)
                #     self.nodetype[node.func] = ty_fun
                #     self.nodetype[node] = ty_fun.retty


        elif isinstance(node, gast.Num):
            # Num(object n)
            if isinstance(node.n, int):
                self.nodetype[node] = T.TyInt()
            elif isinstance(node.n, float):
                self.nodetype[node] = T.TyFloat()


        elif isinstance(node, gast.Str):
            # Str(string s)
            self.nodetype[node] = T.TyString()


        elif isinstance(node, gast.NameConstant):
            # NameConstant(singleton value)
            # value is either True, False or None
            if isinstance(node.value, bool):
                self.nodetype[node] = T.TyBool()
            elif node.value is None:
                self.nodetype[node] = T.TyNone()


        elif isinstance(node, gast.Attribute):
            # Attribute(expr value, identifier attr, expr_context ctx)

            if isinstance(node.value, gast.Name) and \
                    node.value.id in self.args.keys():
                # attributes of arguments (ex. self)
                value = getattr(self.args[node.value.id], node.attr)
                self.nodetype[node] = T.type_of_value(value)

            else:
                ty_obj = self.infer_expr(node.value)
                debug(ty_obj)

                if isinstance(ty_obj, T.TySequence) and ty_obj.is_list():
                    ty_obj.coerce_to_variable_len()
                    self.nodetype[node] = list_attr_ty[node.attr](ty_obj)

                elif isinstance(ty_obj, T.TyUserDefinedClass):
                    # x: value of existing instance
                    x = getattr(ty_obj.instance, node.attr)
                    if callable(x):
                        raise self.ArgumentRequired(ty_obj, x)
                    else:
                        self.nodetype[node] = T.type_of_value(x)


        elif isinstance(node, gast.Subscript):
            # Subscript(expr value, slice slice, expr_context ctx)
            ty_obj = self.infer_expr(node.value)

            if isinstance(ty_obj, T.TySequence):
                self.infer_slice(node.slice, T.TyInt())
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

            elif isinstance(ty_obj, T.TyDict):
                self.infer_slice(node.slice, ty_obj.keyty)
                assert isinstance(node.slice, gast.Index)
                self.nodetype[node] = ty_obj.valty

            elif isinstance(ty_obj, T.TyNdarray):
                self.infer_slice(node.slice, T.TyInt())
                if isinstance(node.slice, gast.Index):
                    self.nodetype[node] = ty_obj.ty
                elif isinstance(node.slice, gast.Slice):
                    self.nodetype[node] = ty_obj
                else:
                    assert False

            else:
                assert False


        elif isinstance(node, gast.Name):
            # Name(identifier id, expr_context ctx, expr? annotation)
            if node.id in self.tyenv.keys():
                self.nodetype[node] = self.tyenv[node.id]
            elif node.id in builtins_name:
                self.nodetype[node] = deepcopy(builtins_ty[eval(node.id)])
            else:
                # case of Tuple assignment
                ty_var = T.TyVar()
                self.tyenv[node.id] = ty_var
                self.nodetype[node] = ty_var


        elif isinstance(node, gast.List):
            # List(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = T.TyList(elts_ty)


        elif isinstance(node, gast.Tuple):
            # Tuple(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = T.TyTuple(elts_ty)


        return self.nodetype[node]


    def infer_slice(self, node: 'gast.Node', ty_key_expected) -> 'NoneType':
        if isinstance(node, gast.Slice):
            # Slice(expr? lower, expr? upper, expr? step)
            if node.lower:
                ty_lower = self.infer_expr(node.lower)
                T.unify(ty_lower, ty_key_expected)
            if node.upper:
                ty_upper = self.infer_expr(node.upper)
                T.unify(ty_upper, ty_key_expected)
            if node.step:
                ty_step = self.infer_expr(node.step)
                T.unify(ty_step, ty_key_expected)
            return

        if isinstance(node, gast.Index):
            # Index(expr value)
            ty_val = self.infer_expr(node.value)
            T.unify(ty_val, ty_key_expected)
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
    except T.UnifyError as e:
        print(traceback.format_exc(), end="")
