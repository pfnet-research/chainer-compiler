import unittest
import ast, gast
from chainer_compiler.elichika.parser import canonicalizer
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import compare_ast


class Break(unittest.TestCase):
    def setUp(self):
        self.canonicalizer = canonicalizer.Canonicalizer()

    def test_break(self):
        orig_code = """
        x = 0
        for i in range(10):
            if i >= 5:
                break
            x += 1
        """
        target_code = """
        x = 0
        for i in range(10):
            if i >= 5:
                breaked_0 = True
            if not breaked_0:
                x+=1
            if breaked_0:
                break
        """
        orig_ast = gast.ast_to_gast(ast.parse(utils.clip_head(orig_code)))
        target_ast = gast.ast_to_gast(ast.parse(utils.clip_head(target_code)))

        assert compare_ast(self.canonicalizer.visit(orig_ast), target_ast)


# ======================================

def main():
    unittest.main()

if __name__ == '__main__': 
    main()