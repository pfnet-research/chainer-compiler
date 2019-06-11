# A module to canonicalize Python AST to a simplified one.
#
# Usage from commandline:
#
# $ python3 elichika/elichika/parser/canonicalizer.py test.py
#

import gast

class Canonicalizer(gast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.for_continue_stack = []
        self.flagid = -1

    def getflag(self):
        self.flagid += 1
        return self.flagid

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, gast.USub) and isinstance(node.operand, gast.Num):
            value = eval(compile(gast.Expression(node), '', 'eval'))
            replacement = gast.Num(n=value)
            return gast.copy_location(replacement, node)
        else:
            return node

    def visit_For(self, node):
        modified_node = self.generic_visit(node)
        continue_flags = self.for_continue_stack.pop()
        for flag in continue_flags:
            node.body.insert(0, gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=False)))
        return modified_node

    def generic_visit(self, node):
        if isinstance(node, gast.stmt):
            if len(self.for_continue_stack) > 0 and len(self.for_continue_stack[-1]) > 0:
                bool_values = []
                for flag in self.for_continue_stack[-1]:
                    bool_values.append(gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=flag, ctx=gast.Load(), annotation=None)))
                if isinstance(node, gast.For):
                    self.for_continue_stack.append([])
                node = super().generic_visit(node)
                if len(bool_values) == 1:
                    cond = bool_values[0]
                else:
                    cond = gast.BoolOp(op=gast.And, values=bool_values)
                replacement = gast.If(test=cond, body=[node], orelse=[])
                ret = gast.copy_location(replacement, node)
            else:
                if isinstance(node, gast.For):
                    self.for_continue_stack.append([])
                ret = super().generic_visit(node)
        else:
            ret = super().generic_visit(node)
        return ret

    def visit_Continue(self, node):
        node = self.generic_visit(node)
        flag = 'continued_' + str(self.getflag())
        self.for_continue_stack[-1].append(flag)
        replacement = gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=True))
        return gast.copy_location(replacement, node)


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
        print(gast.dump(mod))
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

    print('=== Canonicalized AST ===')
    canon_ast = Canonicalizer().visit(orig_ast)
    dump_ast(canon_ast, 'canonicalized')
