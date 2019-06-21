import unittest
import ast, gast
from chainer_compiler.elichika.parser import canonicalizer
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import compare_ast, assert_semantically_equals


class Continue(unittest.TestCase):
    def setUp(self):
        self.canonicalizer = canonicalizer.Canonicalizer(use_illegal_identifier=False)

    def test_continue(self):
        orig_code = utils.clip_head("""
        x = 0
        for i in range(10):
            if i == 5:
                continue
            x += i
        """)
        target_code = utils.clip_head("""
        x = 0
        for i in range(10):
            continued_1 = False
            if i == 5:
                continued_1 = True
            if not continued_1:
                x += i
        """)
        orig_ast = gast.ast_to_gast(ast.parse(orig_code))
        target_ast = gast.ast_to_gast(ast.parse(target_code))
        converted_ast = self.canonicalizer.visit(orig_ast)

        assert_semantically_equals(orig_code, target_code, ['x', 'i'])
        assert compare_ast(converted_ast, target_ast)
        assert compare_ast(target_ast, converted_ast)

    def test_continue_nested(self):
        orig_code = utils.clip_head("""
        x = 0
        for i in range(10):
            if i == 5:
                continue
            for j in range(10):
                if j == 5:
                    continue
                x += i * j
        """)
        target_code = utils.clip_head("""
        x = 0
        for i in range(10):
            continued_1 = False
            if i == 5:
                continued_1 = True
            if not continued_1:
                for j in range(10):
                    continued_2 = False
                    if j == 5:
                        continued_2 = True
                    if not continued_2:
                        x += i * j
        """)
        orig_ast = gast.ast_to_gast(ast.parse(orig_code))
        target_ast = gast.ast_to_gast(ast.parse(target_code))
        converted_ast = self.canonicalizer.visit(orig_ast)

        assert_semantically_equals(orig_code, target_code, ['x', 'i', 'j'])
        assert compare_ast(converted_ast, target_ast)
        assert compare_ast(target_ast, converted_ast)

    def test_continue_break_nested(self):
        orig_code = utils.clip_head("""
        x = 0
        for i in range(10):
            if i == 5:
                continue
            if i == 6:
                break
            for j in range(10):
                if j == 5:
                    break
                x += i * j
        """)
        target_code = utils.clip_head("""
        x = 0
        for i in range(10):
            breaked_1 = False
            continued_1 = False
            if i == 5:
                continued_1 = True
            if not continued_1:
                if i == 6:
                    breaked_1 = True
            if not continued_1 and not breaked_1:
                for j in range(10):
                    breaked_2 = False
                    if j == 5:
                        breaked_2 = True
                    if not breaked_2:
                        x += i * j
                    keepgoing = not breaked_2
                    if breaked_2:
                        break
            keepgoing = not breaked_1
            if breaked_1:
                break
        """)
        orig_ast = gast.ast_to_gast(ast.parse(orig_code))
        target_ast = gast.ast_to_gast(ast.parse(target_code))
        converted_ast = self.canonicalizer.visit(orig_ast)

        assert_semantically_equals(orig_code, target_code, ['x', 'i', 'j'])
        assert compare_ast(converted_ast, target_ast)
        assert compare_ast(target_ast, converted_ast)


# ======================================

def main():
    unittest.main()

if __name__ == '__main__': 
    main() 