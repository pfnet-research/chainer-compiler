import gast

class Type():
    def show(self):
        return ""

class TyUnit(Type):
    def show(self):
        return "()"

class TyInt(Type):
    def show(self):
        return "int"

class TyFloat(Type):
    def show(self):
        return "float"

class TyList(Type):
    def __init__(self, ty):
        super().__init__()
        self.ty = ty

    def show(self):
        return self.ty.show() + " list"


# ==============================================================================

class TypeChecker():
    def __init__(self):
        # type environments
        self.tyenv = {}  # string -> Type (internal type env)
        self.nodetype = {}  # Node -> Type (for elichika to use)


    def dump_nodetype(self):
        for node, ty in self.nodetype.items():
            print(gast.dump(node), " : ", ty.show())


    def infer(self, node : 'ast.Node') -> 'Type':
        """
        Adds local type information to self.tyenv while traversing the AST
        returns: type
        """
        print(gast.dump(node))
        print()

        if isinstance(node, gast.Module):
            self.nodetype[node] = self.infer(node.body[0])
            return

        elif isinstance(node, gast.FunctionDef):
            # TODO(momohatt): Add args to env

            for expr in node.body:
                self.infer(expr)

            # TODO(momohatt): infer function type
            self.nodetype[node] = TyUnit()


        elif isinstance(node, gast.Call):
            if node.func.id == "range":
                assert(len(node.args) <= 3)
                for arg in node.args:
                    assert(isinstance(self.infer(arg), TyInt))
                self.nodetype[node] = TyList(TyInt())

            elif node.func.id == "float":
                assert(len(node.args) == 1)
                # TODO(momohatt): Support string -> float
                assert(isinstance(self.infer(node.args[0]), TyInt))
                self.nodetype[node] = TyFloat()


        elif isinstance(node, gast.Return):
            self.nodetype[node] = self.infer(node.value)


        elif isinstance(node, gast.Assign):
            # TODO(momohatt): support multiple assignment (ex. x, y = 1, 2)
            assert(len(node.targets) == 1)

            var = node.targets[0]
            assert(isinstance(var, gast.Name))
            self.tyenv[var.id] = self.infer(node.value)
            self.nodetype[node] = TyUnit()


        elif isinstance(node, gast.AugAssign):
            # Desugar to BinOp
            self.tyenv[node.target.id] = self.infer(gast.BinOp(node.target, node.op, node.value))
            self.nodetype[node] = TyUnit()


        elif isinstance(node, gast.For):
            # iterate variable
            assert(isinstance(node.target, gast.Name))

            tylist = self.infer(node.iter)
            # TODO(momohatt): Support 'iterator' type
            assert(isinstance(tylist, TyList))
            self.tyenv[node.target.id] = tylist.ty

            for expr in node.body:
                self.infer(expr)

            self.nodetype[node] = TyUnit()


        elif isinstance(node, gast.BinOp):
            # TODO(momohatt): Support overload of operators.

            tyl = self.infer(node.left)
            tyr = self.infer(node.right)
            assert(isinstance(tyl, TyInt) or isinstance(tyl, TyFloat))
            assert(isinstance(tyr, TyInt) or isinstance(tyr, TyFloat))

            if isinstance(node.op, gast.Add) or isinstance(node.op, gast.Sub) or isinstance(node.op, gast.Mul) \
                    or isinstance(node.op, gast.FloorDiv):
                if isinstance(tyl, TyInt) and isinstance(tyr, TyInt):
                    self.nodetype[node] = TyInt()
                else:
                    self.nodetype[node] =  TyFloat()
            elif isinstance(node.op, gast.Div):
                self.nodetype[node] = TyFloat()


        elif isinstance(node, gast.Num):
            if isinstance(node.n, int):
                self.nodetype[node] = TyInt()
            elif isinstance(node.n, float):
                self.nodetype[node] = TyFloat()


        elif isinstance(node, gast.Name):
            self.nodetype[node] = self.tyenv[node.id]

        return self.nodetype[node]

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
