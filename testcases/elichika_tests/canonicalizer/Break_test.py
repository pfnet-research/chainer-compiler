import unittest
import ast, gast
from chainer_compiler.elichika.parser import canonicalizer
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import compare_ast, assert_semantically_equals


class Break(unittest.TestCase):
    def setUp(self):
        self.canonicalizer = canonicalizer.Canonicalizer()

    def test_break(self):
        orig_code = utils.clip_head("""
        x = 0
        for i in range(10):
            if i == 5:
                break
            x += i
        """)
        target_code = utils.clip_head("""
        x = 0
        for i in range(10):
            breaked_0 = False
            if i == 5:
                breaked_0 = True
            if not breaked_0:
                x += i
            keepgoing = not breaked_0
            if breaked_0:
                break
        """)
        orig_ast = gast.ast_to_gast(ast.parse(orig_code))
        target_ast = gast.ast_to_gast(ast.parse(target_code))
        converted_ast = self.canonicalizer.visit(orig_ast)

        assert_semantically_equals(orig_code, target_code, ['x', 'i'])
        assert compare_ast(converted_ast, target_ast)
        assert compare_ast(target_ast, converted_ast)

    def test_break_nested(self):
        orig_code = utils.clip_head("""
        x = 0
        for i in range(10):
            if i == 5:
                break
            for j in range(10):
                if j == 5:
                    break
                x += i * j
        """)
        target_code = utils.clip_head("""
        x = 0
        for i in range(10):
            breaked_0 = False
            if i == 5:
                breaked_0 = True
            if not breaked_0:
                for j in range(10):
                    breaked_1 = False
                    if j == 5:
                        breaked_1 = True
                    if not breaked_1:
                        x += i * j
                    keepgoing = not breaked_1
                    if breaked_1:
                        break
            keepgoing = not breaked_0
            if breaked_0:
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