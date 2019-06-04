# A module to canonicalize Python AST to a simplified one.
#
# Usage from commandline:
#
# $ python3 elichika/elichika/parser/canonicalizer.py test.py
#

import ast, gast
from ast import NodeTransformer

class Canonicalizer(NodeTransformer):

    def __init__(self):
        super().__init__()
        self.for_continue_stack = []
        self.flagid = -1

    def getflag(self):
        self.flagid += 1
        return self.flagid

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Num):
            value = eval(compile(ast.Expression(node), '', 'eval'))
            replacement = ast.Num(n=value)
            return ast.copy_location(replacement, node)
        else:
            return node

    def visit_For(self, node):
        modified_node = self.generic_visit(node)
        continue_flags = self.for_continue_stack.pop()
        for flag in continue_flags:
            node.body.insert(0, ast.Assign(targets=[ast.Name(id=flag, ctx=ast.Store())], value=ast.NameConstant(value=False)))
        return modified_node

    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            if len(self.for_continue_stack) > 0 and len(self.for_continue_stack[-1]) > 1:
                bool_values = []
                for flag in self.for_continue_stack[-1]:
                    bool_values.append(ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=flag, ctx=ast.Load())))
                if isinstance(node, ast.For):
                    self.for_continue_stack.append([])
                node = super().generic_visit(node)
                replacement = ast.If(test=ast.BoolOp(op=ast.And, values=bool_values), body=[node], orelse=[])
                ret = ast.copy_location(replacement, node)
            elif len(self.for_continue_stack) > 0 and len(self.for_continue_stack[-1]) == 1:
                flag = self.for_continue_stack[-1][0]
                if isinstance(node, ast.For):
                    self.for_continue_stack.append([])
                node = super().generic_visit(node)
                replacement = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=flag, ctx=ast.Load())), body=[node], orelse=[])
                ret = ast.copy_location(replacement, node)
            else:
                if isinstance(node, ast.For):
                    self.for_continue_stack.append([])
                ret = super().generic_visit(node)
        else:
            ret = super().generic_visit(node)
        return ret

    def visit_Continue(self, node):
        node = self.generic_visit(node)
        flag = 'continued_' + str(self.getflag())
        self.for_continue_stack[-1].append(flag)
        replacement = ast.Assign(targets=[ast.Name(id=flag, ctx=ast.Store())], value=ast.NameConstant(value=True))
        return ast.copy_location(replacement, node)


if __name__ == '__main__':
    import ast
    import gast
    import sys

    def dump_ast(mod):
        mod = gast.ast_to_gast(mod)
        for b in mod.body:
            print(ast.dump(b))

    code = open(sys.argv[1]).read()
    orig_ast = ast.parse(code)
    print('=== Original AST ===')
    dump_ast(orig_ast)
    print('=== Canonicalized AST ===')
    canon_ast = Canonicalizer().visit(orig_ast)
    dump_ast(canon_ast)
