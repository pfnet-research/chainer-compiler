# Usage from commandline:
#
# $ python3 elichika/parser/typing.py test.py

import gast

class Type():
    pass

class TyUnit(Type):
    def __str__(self):
        return "unit"
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
        return "a" + str(self.i)


class UnifyError(Exception):
    def __init__(self, msg):
        self.msg = msg


primitive_func_ty = {
        # TODO(momohatt): maybe use 'TyUnion' instead of list?
        # (int \/ float) -> float
        float : TyArrow([TyUnion([TyInt(), TyFloat()])], TyFloat()),
        # int -> int \/ int -> int -> int \/ int -> int -> int -> int
        range : [
            TyArrow([TyInt()], TyList(TyInt())),
            TyArrow([TyInt(), TyInt()], TyList(TyInt())),
            TyArrow([TyInt(), TyInt(), TyInt()], TyList(TyInt())),
            ],
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
    def __init__(self):
        # type environments
        self.tyenv = {}  # string -> Type (internal type env)
        self.nodetype = {}  # Node -> Type (for elichika to use)


    def dump_nodetype(self):
        for node, ty in self.nodetype.items():
            print(gast.dump(node) + " : \x1b[36m" + str(ty) + "\x1b[39m")


    def infer(self, node : 'ast.Node') -> 'Type':
        """
        Adds local type information to self.tyenv while traversing the AST
        returns: type
        """
        print(gast.dump(node))
        print()


        if isinstance(node, gast.Module):
            self.nodetype[node] = self.infer(node.body[0])


        elif isinstance(node, gast.FunctionDef):
            # TODO(momohatt): Add args to env

            for expr in node.body:
                ty = self.infer(expr)

            # TODO(momohatt): type of function definition?
            self.nodetype[node] = ty


        elif isinstance(node, gast.Expr):
            self.nodetype[node] = self.infer(node.value)


        elif isinstance(node, gast.Call):
            ty_args = [self.infer(arg) for arg in node.args]
            ty_ret = TyVar()

            if isinstance(node.func, gast.Name) and eval(node.func.id) in primitive_func_ty.keys():
                # case of applying primitive functions
                ty_fun = primitive_func_ty[eval(node.func.id)]
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                ty_ret = deref_type(ty_ret)
                self.nodetype[node.func] = TyArrow(ty_args, ty_ret)
            else:
                ty_fun = self.infer(node.func)
                unify(ty_fun, TyArrow(ty_args, ty_ret))
                ty_ret = deref_type(ty_ret)

            self.nodetype[node] = ty_ret


        elif isinstance(node, gast.Return):
            self.nodetype[node] = self.infer(node.value)


        elif isinstance(node, gast.Assign):
            # TODO(momohatt): support multiple assignment (ex. x, y = 1, 2)
            assert(len(node.targets) == 1)

            var = node.targets[0]
            assert(isinstance(var, gast.Name))
            self.tyenv[var.id] = self.infer(node.value)
            self.nodetype[var] = self.tyenv[var.id]
            self.nodetype[node] = TyUnit()


        elif isinstance(node, gast.AugAssign):
            # Desugar to BinOp
            self.tyenv[node.target.id] = self.infer(gast.BinOp(node.target, node.op, node.value))
            self.nodetype[node] = TyUnit()


        elif isinstance(node, gast.Attribute):
            pass


        elif isinstance(node, gast.For):
            # iterate variable
            # TODO(momohatt): Support cases where node.target is Tuple
            assert(isinstance(node.target, gast.Name))

            tylist = self.infer(node.iter)
            # TODO(momohatt): Support iterator type
            assert(isinstance(tylist, TyList))
            self.tyenv[node.target.id] = tylist.ty

            for expr in node.body:
                self.infer(expr)

            self.nodetype[node] = TyUnit()


        elif isinstance(node, gast.BinOp):
            tyl = self.infer(node.left)
            tyr = self.infer(node.right)

            ty_ops = primitive_op_ty[type(node.op)]
            ty_ret = TyVar()
            unify(ty_ops, TyArrow([tyl, tyr], ty_ret))

            ty_ret = deref_type(ty_ret)
            self.nodetype[node.op] = TyArrow([tyl, tyr], ty_ret)
            self.nodetype[node] = ty_ret


        elif isinstance(node, gast.List):
            if node.elts:
                # Type assertion of list
                elts_ty = [self.infer(e) for e in node.elts]
                assert(all([type(e) == type(elts_ty[0]) for e in elts_ty[1:]]))
                self.nodetype[node] = TyList(elts_ty[0])
            else:
                # Types of empty lists will be determined later
                self.nodetype[node] = TyList(TyVar())


        elif isinstance(node, gast.Num):
            if isinstance(node.n, int):
                self.nodetype[node] = TyInt()
            elif isinstance(node.n, float):
                self.nodetype[node] = TyFloat()


        elif isinstance(node, gast.Name):
            self.nodetype[node] = self.tyenv[node.id]


        return self.nodetype[node]


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
    if isinstance(ty1, TyUnit) and isinstance(ty2, TyUnit):
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
