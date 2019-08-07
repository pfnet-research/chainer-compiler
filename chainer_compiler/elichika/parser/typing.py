# Usage from commandline:
#
# $ python3 elichika/parser/typing.py test.py

from copy import deepcopy
import gast

# TODO(momohatt): rename 'Type' into 'Objects'?
class Type():
    pass

# --------------------------- python primivite types ---------------------------

class TyNone(Type):  # kind of 'unit'
    def __str__(self):
        return "none"
    def __repr__(self):
        return self.__str__()


class TyBool(Type):
    def __str__(self):
        return "bool"
    def __repr__(self):
        return self.__str__()


class TyInt(Type):
    def __str__(self):
        return "int"
    def __repr__(self):
        return self.__str__()


class TyFloat(Type):
    def __str__(self):
        return "float"
    def __repr__(self):
        return self.__str__()


class TyList(Type):
    def __init__(self, ty):
        super().__init__()
        self.ty = ty

    def __str__(self):
        return str(self.ty) + " list"
    def __repr__(self):
        return self.__str__()


class TyTuple(Type):
    def __init__(self, tys):
        super().__init__()
        self.tys = tys

    def __str__(self):
        if len(self.tys) == 0:
            return "()"
        return "(" + "".join([str(t) + ", " for t in self.tys[:-1]]) + str(self.tys[-1]) + ")"
    def __repr__(self):
        return self.__str__()


class TyArrow(Type):
    def __init__(self, argty, retty):
        super().__init__()
        self.argty = argty  # Arguments are uncurried
        self.retty = retty

    def __str__(self):
        return "".join([str(t) + " -> " for t in self.argty]) + str(self.retty)
    def __repr__(self):
        return self.__str__()


counter = 0

class TyVar(Type):
    def __init__(self):
        global counter
        super().__init__()
        self.i = counter
        counter += 1
        self.ty = None

    def __str__(self):
        if self.ty:
            return str(self.ty)
            # return "a" + str(self.i) + "(" + str(self.ty) + ")"
        return "a" + str(self.i)
    def __repr__(self):
        return self.__str__()


class UnifyError(Exception):
    def __init__(self, msg):
        self.msg = msg


def all_same_ty(tys):
    if tys == []:
        return True
    return all([type(e) == type(tys[0]) for e in tys[1:]])


# ==============================================================================

primitive_func_ty = {
        # TODO(momohatt): maybe use 'TyUnion' instead of list?
        # (int \/ float) -> float
        float : TyArrow([[TyInt(), TyFloat()]], TyFloat()),
        # int -> int \/ int -> int -> int \/ int -> int -> int -> int
        # TODO(momohatt): maybe it'd be better to first desugar this function into 3-arg version?
        range : [
            TyArrow([TyInt()], TyList(TyInt())),
            TyArrow([TyInt(), TyInt()], TyList(TyInt())),
            TyArrow([TyInt(), TyInt(), TyInt()], TyList(TyInt())),
            ],
        }


# each value should be a function which takes the type of object ('TyList(sth)' in this case)
list_attr_ty = {
        'append' : lambda ty_obj: TyArrow([ty_obj.ty], TyNone()),
        }


primitive_op_ty = {
        gast.Add : [
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            ],
        gast.Sub : [
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            ],
        gast.Mult : [
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            ],
        gast.Div : [
            # (int \/ float) -> (int \/ float) -> float
            TyArrow([ [TyInt(), TyFloat()], [TyInt(), TyFloat()] ], TyFloat()),
            ],
        gast.FloorDiv : [
            TyArrow([TyInt(), TyInt()], TyInt()),
            TyArrow([TyInt(), TyFloat()], TyFloat()),
            TyArrow([TyFloat(), TyInt()], TyFloat()),
            TyArrow([TyFloat(), TyFloat()], TyFloat()),
            ],
        }

# ==============================================================================

class TypeChecker():
    def __init__(self, tyenv=None):
        if tyenv is None:
            self.tyenv = {}  # string -> Type (internal type env)
        else:
            # TODO(momohatt): heavy?
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
            self.nodetype[node] = self.infer_stmt(node.body[0])
        else:
            assert(False)


    # ================================ stmt ====================================
    def infer_stmt(self, node : 'ast.Node') -> 'Type':
        print(gast.dump(node))
        print()

        if isinstance(node, gast.FunctionDef):
            # _fields: name, args, body, decorator_list, returns(type)
            # TODO(momohatt): Add args to env

            for stmt in node.body:
                ty = self.infer_stmt(stmt)

            # TODO(momohatt): type of function definition?
            self.nodetype[node] = ty


        elif isinstance(node, gast.Return):
            # _fields: value
            self.nodetype[node] = self.infer_expr(node.value)


        elif isinstance(node, gast.Assign):
            # _fields: targets, value
            assert(len(node.targets) == 1)  # cannot think of cases where >= 2
            target = node.targets[0]

            if isinstance(target, gast.Name):
                self.tyenv[target.id] = self.infer_expr(node.value)
                self.nodetype[target] = self.tyenv[target.id]
            elif isinstance(target, gast.Attribute):
                pass
            elif isinstance(target, gast.Tuple):
                ty_target = self.infer_expr(target)
                ty_val = self.infer_expr(node.value)
                unify(ty_target, ty_val)
                for (var, ty) in zip(target.elts, ty_val.tys):
                    self.tyenv[var.id] = ty
                    self.nodetype[var] = ty
            else:
                assert(False)

            self.nodetype[node] = TyNone()


        elif isinstance(node, gast.AugAssign):
            # _fields: target, op, value
            # TODO(momohatt): in-place add is different from BinOp
            # Desugar to BinOp
            self.tyenv[node.target.id] = self.infer_expr(gast.BinOp(node.target, node.op, node.value))
            self.nodetype[node] = TyNone()


        elif isinstance(node, gast.For):
            # _fields: target, iter, body, orelse
            # iterate variable
            # TODO(momohatt): Support cases where node.target is Tuple
            assert(isinstance(node.target, gast.Name))

            tylist = self.infer_expr(node.iter)
            # TODO(momohatt): Support iterator type
            assert(isinstance(tylist, TyList))
            self.tyenv[node.target.id] = tylist.ty

            for stmt in node.body:
                self.infer_stmt(stmt)

            self.nodetype[node] = TyNone()


        elif isinstance(node, gast.If):
            # _fields: test, body, orelse
            ty_test = self.infer_expr(node.test)
            unify(ty_test, TyBool())

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
            # _fields: value
            self.nodetype[node] = self.infer_expr(node.value)


        return self.nodetype[node]


    # ================================= expr ===================================
    def infer_expr(self, node : 'ast.Node') -> 'Type':
        print(gast.dump(node))
        print()

        if isinstance(node, gast.BinOp):
            # _fields: left, op, right
            tyl = self.infer_expr(node.left)
            tyr = self.infer_expr(node.right)

            ty_ops = primitive_op_ty[type(node.op)]
            ty_ret = TyVar()
            unify(ty_ops, TyArrow([tyl, tyr], ty_ret))

            ty_ret = deref_type(ty_ret)
            self.nodetype[node.op] = TyArrow([tyl, tyr], ty_ret)
            self.nodetype[node] = ty_ret


        elif isinstance(node, gast.Call):
            # _fields: func, args, keywords
            ty_args = [self.infer_expr(arg) for arg in node.args]
            ty_ret = TyVar()

            if isinstance(node.func, gast.Name) and eval(node.func.id) in primitive_func_ty.keys():
                # case of applying primitive functions
                ty_fun = primitive_func_ty[eval(node.func.id)]
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                ty_ret = deref_type(ty_ret)
                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
                self.nodetype[node] = ty_ret

            elif isinstance(node.func, gast.Attribute):
                ty_fun = self.infer_expr(node.func)
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                self.nodetype[node.func] = deref_type(ty_fun)
                self.nodetype[node] = deref_type(ty_ret)

            else:
                ty_fun = self.infer_expr(node.func)
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                self.nodetype[node] = deref_type(ty_ret)



        elif isinstance(node, gast.Num):
            # _fields: n
            if isinstance(node.n, int):
                self.nodetype[node] = TyInt()
            elif isinstance(node.n, float):
                self.nodetype[node] = TyFloat()


        elif isinstance(node, gast.NameConstant):
            # _fields: value
            # value is either True, False or None
            if isinstance(node.value, bool):
                self.nodetype[node] = TyBool()
            elif node.value is None:
                self.nodetype[node] = TyNone()


        elif isinstance(node, gast.Attribute):
            # _fields: value, attr, ctx
            ty_obj = self.infer_expr(node.value)
            if isinstance(ty_obj, TyList):
                ty_ret = list_attr_ty[node.attr](ty_obj)
                self.nodetype[node] = ty_ret


        elif isinstance(node, gast.Subscript):
            # _fields: value, slice, ctx
            ty_obj = self.infer_expr(node.value)
            self.infer_slice(node.slice)

            if isinstance(ty_obj, TyList):
                if isinstance(node.slice, gast.Index):
                    self.nodetype[node] = ty_obj.ty
                elif isinstance(node.slice, gast.Slice):
                    self.nodetype[node] = ty_obj

            elif isinstance(ty_obj, TyTuple):
                if isinstance(node.slice, gast.Index):
                    index = node.slice.value
                    # TODO(momohatt): handle cases where index is more complex but still a constant
                    if isinstance(index, gast.Num):
                        self.nodetype[node] = ty_obj.tys[index.n]
                    else:
                        assert(all_same_ty(ty_obj.tys))
                        self.nodetype[node] = ty_obj.tys[0]
                elif isinstance(node.slice, gast.Slice):
                    pass

            else:
                assert(False)


        elif isinstance(node, gast.Name):
            # _fields: id, ctx, annotation
            if node.id in self.tyenv.keys():
                self.nodetype[node] = self.tyenv[node.id]
            else:
                # case of Tuple assignment
                self.nodetype[node] = TyVar()


        elif isinstance(node, gast.List):
            # _fields: elts, ctx
            if node.elts == []:
                # Types of empty lists will be determined later
                self.nodetype[node] = TyList(TyVar())
            else:
                # Type assertion of list
                elts_ty = [self.infer_expr(e) for e in node.elts]
                assert(all_same_ty(elts_ty))
                self.nodetype[node] = TyList(elts_ty[0])


        elif isinstance(node, gast.Tuple):
            # _fields: elts, ctx
            elts_ty = [self.infer_expr(e) for e in node.elts]
            self.nodetype[node] = TyTuple(elts_ty)


        return self.nodetype[node]


    def infer_slice(self, node: 'gast.Node') -> 'NoneType' :
        if isinstance(node, gast.Slice):
            # _fields: lower, upper, step
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
            # _fields: value
            ty_val = self.infer_expr(node.value)
            unify(ty_val, TyInt())

        # we shouldn't have to think about the type of 'slice' itself
        return


def deref_type(ty):
    if isinstance(ty, TyVar):
        return deref_type(ty.ty)
    if ty is None:
        print("\x1b[31muninstantinated type found!!!\x1b[39m")
        return TyInt()
    return ty


def unify(ty1, ty2):
    # ty1 is either Type or list of Type.

    # if ty1 is a list of Type, try unification one by one.
    # returns the type where unification succeeded.
    if type(ty1) is list:
        assert(not ty1 == [])
        assert(isinstance(ty1[0], Type))

        for ty1_ in ty1:
            try:
                unify(ty1_, ty2)
                return
            except UnifyError:
                print("\x1b[33m[LOG] unify error with " + str(ty1_) + " and " + str(ty2) + ". continuing...\x1b[39m")
                continue

        raise UnifyError(str(ty1) + " and " + str(ty2) + " are not unifiable")

    assert(isinstance(ty1, Type))

    # if ty1 is a type, just do normal unification
    if isinstance(ty1, TyNone) and isinstance(ty2, TyNone):
        return
    if isinstance(ty1, TyBool) and isinstance(ty2, TyBool):
        return
    if isinstance(ty1, TyInt) and isinstance(ty2, TyInt):
        return
    if isinstance(ty1, TyFloat) and isinstance(ty2, TyFloat):
        return
    if isinstance(ty1, TyArrow) and isinstance(ty2, TyArrow) and len(ty1.argty) == len(ty2.argty):
        for (at1, at2) in zip(ty1.argty, ty2.argty):
            unify(at1, at2)
        unify(ty1.retty, ty2.retty)
        return
    if isinstance(ty1, TyList) and isinstance(ty2, TyList):
        unify(ty1.ty, ty2.ty)
        return
    if isinstance(ty1, TyTuple) and isinstance(ty2, TyTuple) and len(ty1.tys) == len(ty2.tys):
        for (t1, t2) in zip(ty1.tys, ty2.tys):
            unify(t1, t2)
        return


    if isinstance(ty1, TyVar):
        ty1.ty = ty2
        return

    if isinstance(ty2, TyVar):
        ty2.ty = ty1
        return

    raise UnifyError(str(ty1) + " and " + str(ty2) + " are not unifiable")


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
    tc.infer(orig_ast)
    print('=== Type Environment ===')
    tc.dump_nodetype()
