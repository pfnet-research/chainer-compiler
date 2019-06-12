import unittest
import ast, gast
from chainer_compiler.elichika.parser import canonicalizer
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import compare_ast, assert_semantically_equals


class Continue(unittest.TestCase):
    def setUp(self):
        self.canonicalizer = canonicalizer.Canonicalizer()

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
            continued_0 = False
            if i == 5:
                continued_0 = True
            if not continued_0:
                x += i
        """)
        orig_ast = gast.ast_to_gast(ast.parse(orig_code))
        target_ast = gast.ast_to_gast(ast.parse(target_code))

        assert_semantically_equals(orig_code, target_code, ['x', 'i'])
        assert compare_ast(self.canonicalizer.visit(orig_ast), target_ast)


# ======================================

def main():
    unittest.main()

if __name__ == '__main__': 
    main() 