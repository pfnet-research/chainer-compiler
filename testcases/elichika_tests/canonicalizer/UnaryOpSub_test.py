import unittest
import ast, gast
from chainer_compiler.elichika.parser import canonicalizer
from chainer_compiler.elichika.testtools import compare_ast


class UnaryOpSub(unittest.TestCase):
    def setUp(self):
        self.canonicalizer = canonicalizer.Canonicalizer()

    def test_usub(self):
        orig_ast = gast.ast_to_gast(ast.parse("-3"))
        target_ast = gast.Module(body=[gast.Expr(value=gast.Constant(value=-3, kind=None))], type_ignores=[])
        assert compare_ast(self.canonicalizer.visit(orig_ast), target_ast)

# ======================================

def main():
    unittest.main()

if __name__ == '__main__': 
    main() 
