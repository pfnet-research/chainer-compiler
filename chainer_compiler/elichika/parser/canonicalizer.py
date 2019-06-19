# A module to canonicalize Python AST to a simplified one.
#
# Usage from commandline:
#
# $ python3 elichika/elichika/parser/canonicalizer.py test.py
#

import gast

class Canonicalizer(gast.NodeTransformer):

    def __init__(self, use_illegal_identifier=True):
        super().__init__()
        if use_illegal_identifier:
            self.keepgoing_flag = '#keepgoing'
            self.breaked_flag = '#breaked_'
            self.continued_flag = '#continued_'
            self.returned_flag = '#returned_'
            self.returned_value_key = '#returned_value'
        else:
            self.keepgoing_flag = 'keepgoing'
            self.breaked_flag = 'breaked_'
            self.continued_flag = 'continued_'
            self.returned_flag = 'returned_'
            self.returned_value_key = 'returned_value'
        self.for_continue_stack = []
        self.for_breaked_stack = []
        self.func_returned_stack = []
        self.flagid = -1

    def stack_has_flags(self, stack):
        assert isinstance(stack, list)
        return len(stack) > 0 and len(stack[-1]) > 0

    def getflag(self):
        self.flagid += 1
        return self.flagid

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, gast.USub) and isinstance(node.operand, gast.Num):
            value = node.operand.n
            replacement = gast.Num(n=-value)
            return gast.copy_location(replacement, node)
        else:
            return node

    def visit_FunctionDef(self, node):
        modified_node = self.generic_visit(node)
        returned_flags = self.func_returned_stack.pop()
        for flag in returned_flags:
            node.body.insert(0, gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=False)))
        if isinstance(node, gast.FunctionDef):
            node.body.append(gast.Return(value=gast.Name(id=self.returned_value_key, ctx=gast.Load(), annotation=None)))
        return modified_node

    def visit_For(self, node):
        modified_node = self.generic_visit(node)
        continue_flags = self.for_continue_stack.pop()
        for flag in continue_flags:
            node.body.insert(0, gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=False)))
        breaked_flags = self.for_breaked_stack.pop()
        bool_values = []
        for flag in breaked_flags:
            node.body.insert(0, gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=False)))
            bool_values.append(gast.Name(id=flag, ctx=gast.Load(), annotation=None))

        if self.stack_has_flags(self.func_returned_stack):
            returned_flags = self.func_returned_stack[-1]
            for flag in returned_flags:
                bool_values.append(gast.Name(id=flag, ctx=gast.Load(), annotation=None))

        if len(bool_values) > 0:
            if len(bool_values) == 1:
                cond = bool_values[0]
            elif len(bool_values) > 1:
                cond = gast.BoolOp(op=gast.Or(), values=bool_values)
            if isinstance(node, gast.For):
                node.body.append(gast.Assign(targets=[gast.Name(id=self.keepgoing_flag, ctx=gast.Store(), annotation=None)], value=gast.UnaryOp(op=gast.Not(), operand=cond)))
                node.body.append(gast.If(test=cond, body=[gast.Break()], orelse=[]))
        return modified_node

    def generic_visit(self, node):
        if isinstance(node, gast.stmt):
            if self.stack_has_flags(self.for_continue_stack) or self.stack_has_flags(self.for_breaked_stack) or self.stack_has_flags(self.func_returned_stack):
                bool_values = []
                if self.stack_has_flags(self.for_continue_stack):
                    for flag in self.for_continue_stack[-1]:
                        bool_values.append(gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=flag, ctx=gast.Load(), annotation=None)))
                if self.stack_has_flags(self.for_breaked_stack):
                    for flag in self.for_breaked_stack[-1]:
                        bool_values.append(gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=flag, ctx=gast.Load(), annotation=None)))
                if self.stack_has_flags(self.func_returned_stack):
                    for flag in self.func_returned_stack[-1]:
                        bool_values.append(gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=flag, ctx=gast.Load(), annotation=None)))

                if isinstance(node, gast.For):
                    self.for_continue_stack.append([])
                    self.for_breaked_stack.append([])
                elif isinstance(node, gast.FunctionDef):
                    self.func_returned_stack.append([])

                modified_node = super().generic_visit(node)
                if len(bool_values) == 1:
                    cond = bool_values[0]
                else:
                    cond = gast.BoolOp(op=gast.And(), values=bool_values)
                replacement = gast.If(test=cond, body=[modified_node], orelse=[])
                ret = gast.copy_location(replacement, node)
            else:
                if isinstance(node, gast.For):
                    self.for_continue_stack.append([])
                    self.for_breaked_stack.append([])
                elif isinstance(node, gast.FunctionDef):
                    self.func_returned_stack.append([])
                ret = super().generic_visit(node)
        else:
            ret = super().generic_visit(node)
        return ret

    def visit_Return(self, node):
        modified_node = self.generic_visit(node)
        if node.value is None:
            node_value = gast.NameConstant(value=None)
        else:
            node_value = node.value
        flag = self.returned_flag + str(self.getflag())
        self.func_returned_stack[-1].append(flag)
        if isinstance(modified_node, gast.If):
            modified_node.body.clear()
            modified_node.body.append(gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=True)))
            modified_node.body.append(gast.Assign(targets=[gast.Name(id=self.returned_value_key, ctx=gast.Store(), annotation=None)], value=node_value))
            replacement = modified_node
        else:
            replacement  = [gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=True)),
                            gast.Assign(targets=[gast.Name(id=self.returned_value_key, ctx=gast.Store(), annotation=None)], value=node_value)]
        return replacement  #TODO: Add location to replacement


    def visit_Continue(self, node):
        modified_node = self.generic_visit(node)
        flag = self.continued_flag + str(self.getflag())
        self.for_continue_stack[-1].append(flag)
        replacement = gast.Assign(targets=[gast.Name(id=flag, ctx=gast.Store(), annotation=None)], value=gast.NameConstant(value=True))
        return gast.copy_location(replacement, node)


    def visit_Break(self, node):
        modified_node = self.generic_visit(node)
        flag = self.breaked_flag + str(self.getflag())
        self.for_breaked_stack[-1].append(flag)
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
