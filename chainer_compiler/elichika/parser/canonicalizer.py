# A module to canonicalize Python AST to a simplified one.
#
# Usage from commandline:
#
# $ python3 elichika/elichika/parser/canonicalizer.py test.py
#

import gast
import numbers

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
        self.for_continued_stack = []
        self.for_breaked_stack = []
        self.func_returned_stack = []
        self.flagid = -1

    def get_id(self, stack):
        return len(stack)

    def stack_has_flags(self, stack):
        assert isinstance(stack, list)
        return len(stack) > 0 and stack[-1]

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, gast.USub) and (isinstance(node.operand, gast.Constant) and isinstance(node.operand.value, numbers.Number)):
            value = node.operand.value
            replacement = gast.Constant(value=-value, kind=None)
            return gast.copy_location(replacement, node)
        else:
            return node

    def visit_FunctionDef(self, node):
        modified_node = self.generic_visit(node)
        returned_id = len(self.func_returned_stack)
        returned_flags = self.func_returned_stack.pop()
        if returned_flags:
            node.body.insert(0, gast.Assign(targets=[gast.Name(id=self.returned_flag + str(returned_id), ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(value=False, kind=None)))
        node.body.insert(0, gast.Assign(targets=[gast.Name(id=self.returned_value_key, ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(value=None, kind=None)))
        node.body.append(gast.Return(value=gast.Name(id=self.returned_value_key, ctx=gast.Load(), annotation=None, type_comment=None)))
        return modified_node

    def visit_For(self, node):
        modified_node = self.generic_visit(node)
        continued_id = len(self.for_continued_stack)
        continued_flags = self.for_continued_stack.pop()
        if continued_flags:
            node.body.insert(0, gast.Assign(targets=[gast.Name(id=self.continued_flag + str(continued_id), ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(value=False, kind=None)))
        breaked_id = len(self.for_breaked_stack)
        breaked_flags = self.for_breaked_stack.pop()
        bool_values = []
        if breaked_flags:
            node.body.insert(0, gast.Assign(targets=[gast.Name(id=self.breaked_flag + str(breaked_id), ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(value=False, kind=None)))
            bool_values.append(gast.Name(id=self.breaked_flag + str(breaked_id), ctx=gast.Load(), annotation=None, type_comment=None))

        if len(self.func_returned_stack) > 0:
            returned_id = len(self.func_returned_stack)
            returned_flags = self.func_returned_stack[-1]
            if returned_flags:
                bool_values.append(gast.Name(id=self.returned_flag + str(returned_id), ctx=gast.Load(), annotation=None, type_comment=None))

        if len(bool_values) > 0:
            if len(bool_values) == 1:
                cond = bool_values[0]
            elif len(bool_values) > 1:
                cond = gast.BoolOp(op=gast.Or(), values=bool_values)
            
            node.body.append(gast.Assign(targets=[gast.Name(id=self.keepgoing_flag, ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.UnaryOp(op=gast.Not(), operand=cond)))
            node.body.append(gast.If(test=cond, body=[gast.Break()], orelse=[]))
        return modified_node

    def generic_visit(self, node):
        if isinstance(node, gast.stmt):
            if self.stack_has_flags(self.for_continued_stack) or self.stack_has_flags(self.for_breaked_stack) or self.stack_has_flags(self.func_returned_stack):
                bool_values = []
                if self.stack_has_flags(self.for_continued_stack):
                    continued_id = len(self.for_continued_stack)
                    bool_values.append(gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=self.continued_flag + str(continued_id), ctx=gast.Load(), annotation=None, type_comment=None)))
                if self.stack_has_flags(self.for_breaked_stack):
                    breaked_id = len(self.for_breaked_stack)
                    bool_values.append(gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=self.breaked_flag + str(breaked_id), ctx=gast.Load(), annotation=None, type_comment=None)))
                if self.stack_has_flags(self.func_returned_stack):
                    returned_id = len(self.func_returned_stack)
                    bool_values.append(gast.UnaryOp(op=gast.Not(), operand=gast.Name(id=self.returned_flag + str(returned_id), ctx=gast.Load(), annotation=None, type_comment=None)))

                if isinstance(node, gast.For):
                    self.for_continued_stack.append(False)
                    self.for_breaked_stack.append(False)
                elif isinstance(node, gast.FunctionDef):
                    self.func_returned_stack.append(False)

                modified_node = super().generic_visit(node)
                if len(bool_values) == 1:
                    cond = bool_values[0]
                else:
                    cond = gast.BoolOp(op=gast.And(), values=bool_values)
                replacement = gast.If(test=cond, body=[modified_node], orelse=[])
                ret = gast.copy_location(replacement, node)
            else:
                if isinstance(node, gast.For):
                    self.for_continued_stack.append(False)
                    self.for_breaked_stack.append(False)
                elif isinstance(node, gast.FunctionDef):
                    self.func_returned_stack.append(False)
                ret = super().generic_visit(node)
        else:
            ret = super().generic_visit(node)
        return ret

    def visit_Return(self, node):
        modified_node = self.generic_visit(node)
        if node.value is None:
            node_value = gast.Constant(value=None, kind=None)
        else:
            node_value = node.value
        self.func_returned_stack[-1] = True
        returned_id = len(self.func_returned_stack)
        replacement  = [gast.Assign(targets=[gast.Name(id=self.returned_flag + str(returned_id), ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(value=True, kind=None)),
                        gast.Assign(targets=[gast.Name(id=self.returned_value_key, ctx=gast.Store(), annotation=None, type_comment=None)], value=node_value)]
        if isinstance(modified_node, gast.If):  #TODO: Add location to returned value.
            modified_node.body = replacement
            return modified_node
        else:
            return replacement  

    def visit_Continue(self, node):
        modified_node = self.generic_visit(node)
        self.for_continued_stack[-1] = True
        continued_id = len(self.for_continued_stack)
        replacement = gast.Assign(targets=[gast.Name(id=self.continued_flag + str(continued_id), ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(value=True, kind=None))
        return gast.copy_location(replacement, node)


    def visit_Break(self, node):
        modified_node = self.generic_visit(node)
        self.for_breaked_stack[-1] = True
        breaked_id = len(self.for_breaked_stack)
        replacement = gast.Assign(targets=[gast.Name(id=self.breaked_flag + str(breaked_id), ctx=gast.Store(), annotation=None, type_comment=None)], value=gast.Constant(value=True, kind=None))
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
