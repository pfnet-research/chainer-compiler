# Usage from commandline:
#
# $ python3 elichika/parser/typing.py test.py

from copy import deepcopy
from enum import Enum
import gast

# TODO(momohatt): rename 'Type' into 'Objects'?
class Type():
    pass

# --------------------------- python primivite types ---------------------------

class TyNone(Type):
    def __str__(self):
        return "none"
    def __repr__(self):
        return self.__str__()
    def freeze(self):
        return
    def deref(self):
        return self


class TyBool(Type):
    def __str__(self):
        return "bool"
    def __repr__(self):
        return self.__str__()
    def freeze(self):
        return
    def deref(self):
        return self


class TyInt(Type):
    def __str__(self):
        return "int"
    def __repr__(self):
        return self.__str__()
    def freeze(self):
        return
    def deref(self):
        return self


class TyFloat(Type):
    def __str__(self):
        return "float"
    def __repr__(self):
        return self.__str__()
    def freeze(self):
        return
    def deref(self):
        return self


class TyString(Type):
    def __str__(self):
        return "string"
    def __repr__(self):
        return self.__str__()
    def freeze(self):
        return
    def deref(self):
        return self


class TyArrow(Type):
    def __init__(self, argty, retty):
        super().__init__()
        self.argty = argty  # Arguments are uncurried
        self.retty = retty

    def __str__(self):
        return "{} -> {}".format(self.argty, self.retty)
    def __repr__(self):
        return self.__str__()

    def freeze(self):
        for t in self.argty:
            t.freeze()
        self.retty.freeze()

    def deref(self):
        self.argty = [t.deref() for t in self.argty]
        self.retty = self.retty.deref()
        return self


class SequenceKind(Enum):
    LIST = 0
    TUPLE = 1

class TySequence(Type):
    def __init__(self, seq_kind, ty):
        super().__init__()
        self.seq_kind = seq_kind
        self.is_fixed_len = isinstance(ty, list)
        self.ty_ = ty

    def __str__(self):
        if self.is_fixed_len:
            if self.seq_kind == SequenceKind.LIST:
                return str(self.ty_)

            if self.seq_kind == SequenceKind.TUPLE:
                if len(self.ty_) == 0:
                    return "()"
                return "(" + "".join([str(t) + ", " for t in self.ty_[:-1]]) + str(self.ty_[-1]) + ")"

        if self.seq_kind == SequenceKind.LIST:
            return str(self.ty_) + " list"
        if self.seq_kind == SequenceKind.TUPLE:
            return str(self.ty_) + " tuple"


    def __repr__(self):
        return self.__str__()

    def freeze(self):
        if self.is_fixed_len:
            for t in self.ty_:
                t.freeze()
            return
        self.ty_.freeze()

    def deref(self):
        if self.is_fixed_len:
            self.ty_ = [t.deref() for t in self.ty_]
        else:
            self.ty_ = self.ty_.deref()
        return self

    def get_ty(self):
        assert(not self.is_fixed_len)
        return self.ty_

    def get_tys(self):
        assert(self.is_fixed_len)
        return self.ty_

    def coerce_to_variable_len(self, ty):
        self.ty_ = ty
        self.is_fixed_len = False
        return

    def is_list(self):
        return self.seq_kind == SequenceKind.LIST

    def is_tuple(self):
        return self.seq_kind == SequenceKind.TUPLE


def TyList(ty):  # shorthand notation
    return TySequence(SequenceKind.LIST, ty)

def TyTuple(ty):  # shorthand notation
    return TySequence(SequenceKind.TUPLE, ty)


class TyDict(Type):
    def __init__(self, keyty, valty):
        super().__init__()
        self.keyty = keyty
        self.valty = valty
        pass

    def __str__(self):
        return "{{}:{}}".format(self.keyty, self.valty)
    def __repr__(self):
        return self.__str__()

    def freeze(self):
        self.keyty.freeze()
        self.valty.freeze()

    def deref(self):
        self.keyty = self.keyty.deref()
        self.valty = self.valty.deref()
        return self


counter = 0

class TyVar(Type):
    def __init__(self):
        global counter
        super().__init__()
        self.i = counter
        counter += 1
        self.ty = None
        self.is_frozen= False

    def __str__(self):
        if self.ty:
            return "a{}({})".format(self.i, self.ty)
        return "a" + str(self.i)
    def __repr__(self):
        return self.__str__()

    def freeze(self):
        if self.ty is not None:
            self.is_frozen = True
            self.ty.freeze()

    def deref(self):
        if self.is_frozen:
            return self.ty.deref()
        return self


class TyUnion(Type):
    def __init__(self, *tys):
        assert(len(tys) >= 2)
        self.tys = list(tys)  # tys : tuple of Type
    def __str__(self):
        return str(self.tys[0]) + "".join([" \/ " + str(t) for t in self.tys[1:]])
    def __repr__(self):
        return self.__str__()

    def freeze(self):
        for t in self.tys:
            t.freeze()
    def deref(self):
        self.tys = [t.deref() for t in self.tys]
        return self


class UnifyError(Exception):
    def __init__(self, ty1, ty2):
        self.msg = "UnifyError: {} and {} are not unifiable".format(ty1, ty2)


def all_same_ty(tys):
    if tys == []:
        return True
    return all([type(e) == type(tys[0]) for e in tys[1:]])


# ==============================================================================

primitive_func_ty = {
        # (int \/ float) -> float
        float : TyArrow([TyUnion(TyInt(), TyFloat())], TyFloat()),
        # int -> int \/ int -> int -> int \/ int -> int -> int -> int
        range : TyUnion(
            TyArrow([TyInt()], TyList(TyInt())),
            TyArrow([TyInt(), TyInt()], TyList(TyInt())),
            TyArrow([TyInt(), TyInt(), TyInt()], TyList(TyInt())),
            ),
        abs : TyUnion(
            TyArrow([TyInt()], TyInt()),
            TyArrow([TyFloat()], TyFloat()),
            ),
        }


list_attr_ty = {
        'append'  : lambda ty_obj: TyArrow([ty_obj.get_ty()], TyNone()),
        'reverse' : lambda ty_obj: TyArrow([ty_obj], TyNone()),
        }


primitive_op_ty = {
        # TODO(momohatt): Support '+' of list and tuple
        gast.Add : TyUnion(
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            TyArrow([TyString(), TyString()], TyString()),
            ),
        gast.Sub : TyUnion(
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            ),
        gast.Mult : TyUnion(
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            ),
        gast.Div :
            # (int \/ float) -> (int \/ float) -> float
            TyArrow([TyUnion(TyInt(), TyFloat()), TyUnion(TyInt(), TyFloat())], TyFloat()),
        gast.FloorDiv : TyUnion(
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            ),
        }

# ==============================================================================

class TypeChecker():
    def __init__(self, tyenv=None):
        if tyenv is None:
            self.tyenv = {}  # string -> Type (internal type env)
        else:
            self.tyenv = deepcopy(tyenv)
        # type environments
        self.nodetype = {}  # Node -> Type (for elichika to use)


    def dump_tyenv(self):
        for name, ty in self.tyenv.items():
            print(name + " : \x1b[35m" + str(ty) + "\x1b[39m")


    def dump_nodetype(self):
        for node, ty in self.nodetype.items():
            print(gast.dump(node) + " : \x1b[36m" + str(ty) + "\x1b[39m")


    def infer(self, node : 'ast.Node') -> 'Type':
        """
        Adds local type information to self.tyenv while traversing the AST
        returns: type
        """
        self.infer_mod(node)

    # ================================ mod =====================================
    def infer_mod(self, node : 'ast.Node') -> 'Type':
        print(gast.dump(node))
        print()
        if isinstance(node, gast.Module):
            self.infer_stmt(node.body[0])
        else:
            assert(False)


    # ================================ stmt ====================================
    def infer_stmt(self, node : 'ast.Node') -> 'Type':
        print(gast.dump(node))
        print()

        if isinstance(node, gast.FunctionDef):
            # FunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list, expr? returns)
            # TODO(momohatt): Add args to env

            for stmt in node.body:
                ty = self.infer_stmt(stmt)

            # TODO(momohatt): type of function definition?
            self.nodetype[node] = ty


        elif isinstance(node, gast.Return):
            # Return(expr? value)
            self.nodetype[node] = self.infer_expr(node.value)


        elif isinstance(node, gast.Assign):
            # Assign(expr* targets, expr value)
            assert(len(node.targets) == 1)  # cannot think of cases where >= 2
            target = node.targets[0]

            if isinstance(target, gast.Name):
                self.tyenv[target.id] = self.infer_expr(node.value)
                self.nodetype[target] = self.tyenv[target.id]
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
                assert(False)

            self.nodetype[node] = TyNone()


        elif isinstance(node, gast.AugAssign):
            # AugAssign(expr target, operator op, expr value)
            # TODO(momohatt): in-place add is different from BinOp
            # Desugar to BinOp
            self.tyenv[node.target.id] = self.infer_expr(gast.BinOp(node.target, node.op, node.value))
            self.nodetype[node] = TyNone()


        elif isinstance(node, gast.For):
            # For(expr target, expr iter, stmt* body, stmt* orelse)
            assert(type(node.target) in [gast.Name, gast.Tuple])

            ty_iteration = self.infer_expr(node.iter)
            ty_i = self.infer_expr(node.target)
            unify(ty_iteration, TyList(ty_i))
            if ty_iteration.is_fixed_len:
                ty_iteration.coerce_to_variable_len(ty_i)

            for stmt in node.body:
                self.infer_stmt(stmt)

            self.nodetype[node] = TyNone()


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
                for name, ty in tc.tyenv.items():
                    unify(ty, self.tyenv[name])

            else:
                tc1 = TypeChecker(self.tyenv)
                tc2 = TypeChecker(self.tyenv)
                for stmt in node.body:
                    tc1.infer_stmt(stmt)
                for stmt in node.orelse:
                    tc2.infer_stmt(stmt)

                # unify the intersection of 2 tyenvs
                for name, ty in tc1.tyenv.items():
                    if name not in tc2.tyenv.keys():
                        continue
                    unify(ty, tc2.tyenv[name])  # untypeable If-stmts will raise error here

                # update local tyenv
                for name, ty in tc1.tyenv.items():
                    if name not in tc2.tyenv.keys():
                        continue
                    self.tyenv[name] = ty

                # merge nodetype from 2 TypeCheckers
                for node_, ty in tc1.nodetype.items():
                    self.nodetype[node_] = ty
                for node_, ty in tc2.nodetype.items():
                    self.nodetype[node_] = ty

            self.nodetype[node] = TyNone()


        elif isinstance(node, gast.Expr):
            # Expr(expr value)
            self.nodetype[node] = self.infer_expr(node.value)


        return self.nodetype[node]


    # ================================= expr ===================================
    def infer_expr(self, node : 'ast.Node') -> 'Type':
        print(gast.dump(node))
        print()

        if isinstance(node, gast.BoolOp):
            # BoolOp(boolop op, expr* values)
            ty_vals = [self.infer_expr(val) for val in node.values]
            for ty in ty_vals:
                unify(ty, TyBool())
            self.nodetype[node.op] = TyArrow([TyBool(), TyBool()], TyBool())  # probably only this type?
            self.nodetype[node] = TyBool()


        elif isinstance(node, gast.BinOp):
            # BinOp(expr left, operator op, expr right)
            tyl = self.infer_expr(node.left)
            tyr = self.infer_expr(node.right)

            ty_ops = primitive_op_ty[type(node.op)]
            ty_ret = TyVar()
            unify(ty_ops, TyArrow([tyl, tyr], ty_ret))

            ty_ret = ty_ret.deref()
            self.nodetype[node.op] = TyArrow([tyl, tyr], ty_ret)
            self.nodetype[node] = ty_ret


        elif isinstance(node, gast.UnaryOp):
            # UnaryOp(unaryop op, expr operand)
            pass


        elif isinstance(node, gast.Dict):
            # Dict(expr* keys, expr* values)
            if node.keys == []:
                self.nodetype[node] = TyDict(TyVar(), TyVar())
            else:
                ty_keys = [self.infer_expr(key) for key in node.keys]
                ty_vals = [self.infer_expr(val) for val in node.values]
                # TODO(momohatt): unify here
                assert(all_same_ty(ty_keys))
                assert(all_same_ty(ty_vals))
                self.nodetype[node] = TyDict(ty_keys[0], ty_vals[0])


        elif isinstance(node, gast.Compare):
            # Compare(expr left, cmpop* ops, expr* comparators)
            pass


        elif isinstance(node, gast.Call):
            # Call(expr func, expr* args, keyword* keywords)
            ty_args = [self.infer_expr(arg) for arg in node.args]
            ty_ret = TyVar()

            if isinstance(node.func, gast.Name) and eval(node.func.id) in primitive_func_ty.keys():
                # case of applying primitive functions
                ty_fun = primitive_func_ty[eval(node.func.id)]
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                ty_ret = ty_ret.deref()
                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
                self.nodetype[node] = ty_ret

            elif isinstance(node.func, gast.Attribute):
                ty_fun = self.infer_expr(node.func)
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                self.nodetype[node.func] = ty_fun.deref()
                self.nodetype[node] = ty_ret.deref()

            else:
                ty_fun = self.infer_expr(node.func)
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                self.nodetype[node] = ty_ret.deref()


        elif isinstance(node, gast.Num):
            # Num(object n)
            if isinstance(node.n, int):
                self.nodetype[node] = TyInt()
            elif isinstance(node.n, float):
                self.nodetype[node] = TyFloat()


        elif isinstance(node, gast.Str):
            # Str(string s)
            self.nodetype[node] = TyString()


        elif isinstance(node, gast.NameConstant):
            # NameConstant(singleton value)
            # value is either True, False or None
            if isinstance(node.value, bool):
                self.nodetype[node] = TyBool()
            elif node.value is None:
                self.nodetype[node] = TyNone()


        elif isinstance(node, gast.Attribute):
            # Attribute(expr value, identifier attr, expr_context ctx)
            ty_obj = self.infer_expr(node.value)
            if ty_obj.is_list():
                if ty_obj.is_fixed_len:  # if the object is fixed-length list, coerce it to variable-length
                    unify(ty_obj, TyList(TyVar()))
                ty_ret = list_attr_ty[node.attr](ty_obj)
                self.nodetype[node] = ty_ret


        elif isinstance(node, gast.Subscript):
            # Subscript(expr value, slice slice, expr_context ctx)
            ty_obj = self.infer_expr(node.value)
            self.infer_slice(node.slice)

            if isinstance(ty_obj, TySequence):
                if ty_obj.is_fixed_len:
                    if isinstance(node.slice, gast.Index) and isinstance(node.slice.value, gast.Num):
                        # TODO(momohatt): handle cases where index is more complex but still a constant
                        self.nodetype[node] = ty_obj.get_tys()[node.slice.value.n]
                    else:
                        ty_elt = TyVar()
                        unify(ty_obj, TyList(ty_elt))
                        if isinstance(node.slice, gast.Index):
                            self.nodetype[node] = ty_elt
                        elif isinstance(node.slice, gast.Slice):
                            self.nodetype[node] = ty_obj

                else:
                    if isinstance(node.slice, gast.Index):
                        self.nodetype[node] = ty_obj.get_ty()
                    elif isinstance(node.slice, gast.Slice):
                        self.nodetype[node] = ty_obj
                    else:
                        assert(False)
            else:
                ty_elt = TyVar()
                unify(ty_obj, TyList(ty_elt))
                if isinstance(node.slice, gast.Index):
                    self.nodetype[node] = ty_elt
                elif isinstance(node.slice, gast.Slice):
                    self.nodetype[node] = TyList(ty_elt)

            assert(self.nodetype[node])


        elif isinstance(node, gast.Name):
            # Name(identifier id, expr_context ctx, expr? annotation)
            if node.id in self.tyenv.keys():
                self.nodetype[node] = self.tyenv[node.id]
            else:
                # case of Tuple assignment
                ty_var = TyVar()
                self.tyenv[node.id] = ty_var
                self.nodetype[node] = ty_var


        elif isinstance(node, gast.List):
            # List(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyList(elts_ty)


        elif isinstance(node, gast.Tuple):
            # Tuple(expr* elts, expr_context ctx)
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyTuple(elts_ty)


        return self.nodetype[node]


    def infer_slice(self, node: 'gast.Node') -> 'NoneType' :
        if isinstance(node, gast.Slice):
            # Slice(expr? lower, expr? upper, expr? step)
            if node.lower:
                ty_lower = self.infer_expr(node.lower)
                unify(ty_lower, TyInt())
            if node.upper:
                ty_upper = self.infer_expr(node.upper)
                unify(ty_upper, TyInt())
            if node.step:
                ty_step = self.infer_expr(node.step)
                unify(ty_step, TyInt())


        elif isinstance(node, gast.Index):
            # Index(expr value)
            ty_val = self.infer_expr(node.value)
            unify(ty_val, TyInt())

        # we shouldn't have to think about the type of 'slice' itself
        return


def unify(ty1, ty2):
    unify_(ty1, ty2)
    if not isinstance(ty1, TyUnion):
        ty1.freeze()
    ty2.freeze()


def unify_(ty1, ty2):
    # ty1 is either Type or list of Type.
    # ty2 must be Type.

    # if ty1 is TyUnion, try unification one by one.
    if isinstance(ty1, TyUnion):
        for ty1_ in ty1.tys:
            try:
                unify_(ty1_, ty2)
                ty1_.freeze()  # not necessary?
                ty2.freeze()
                return
            except UnifyError:
                print("\x1b[33m[LOG] unify error with " + str(ty1_) + " and " + str(ty2) + ". continuing...\x1b[39m")
                continue

        raise UnifyError(ty1, ty2)

    ty1 = ty1.deref()
    ty2 = ty2.deref()

    # if ty1 is not TyUnion, just do normal unification
    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return
    if isinstance(ty1, TyBool) and isinstance(ty2, TyBool):
        return
    if isinstance(ty1, TyInt) and isinstance(ty2, TyInt):
        return
    if isinstance(ty1, TyFloat) and isinstance(ty2, TyFloat):
        return
    if isinstance(ty1, TyString) and isinstance(ty2, TyString):
        return
    if isinstance(ty1, TyArrow) and isinstance(ty2, TyArrow) and len(ty1.argty) == len(ty2.argty):
        for (at1, at2) in zip(ty1.argty, ty2.argty):
            unify_(at1, at2)
        unify_(ty1.retty, ty2.retty)
        return

    if isinstance(ty1, TySequence) and isinstance(ty2, TySequence):
        if ty1.is_fixed_len and ty2.is_fixed_len:
            if not len(ty1.get_tys()) == len(ty2.get_tys()):
                raise UnifyError(ty1, ty2)
            for (t1, t2) in zip(ty1.get_tys(), ty2.get_tys()):
                unify_(t1, t2)
            return
        if ty1.is_fixed_len and not ty2.is_fixed_len:
            for ty in ty1.get_tys():
                unify_(ty, ty2.get_ty())
            ty1.coerce_to_variable_len(ty2.get_ty())
            return
        if (not ty1.is_fixed_len) and ty2.is_fixed_len:
            unify_(ty2, ty1)
            return
        unify_(ty2.get_ty(), ty1.get_ty())
        return

    if isinstance(ty1, TyVar):
        assert(not ty1.is_frozen)
        ty1.ty = ty2
        return

    if isinstance(ty2, TyVar):
        assert(not ty2.is_frozen)
        ty2.ty = ty1
        return

    raise UnifyError(ty1, ty2)


if __name__ == '__main__':
    import ast
    import gast
    import sys
    from copy import deepcopy

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
    print('=== Original AST ===')
    dump_ast(orig_ast, 'original')
    print()

    tc = TypeChecker()
    try:
        tc.infer(orig_ast)
        print('=== Type Environment ===')
        tc.dump_nodetype()
    except UnifyError as e:
        print("\x1b[31m" + e.msg + "\x1b[39m")
