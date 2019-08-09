import ast, gast
import pprint
import unittest

from chainer_compiler.elichika.parser import typing
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import generate_id_table, generate_type_table


class TestFor(unittest.TestCase):
    def test_hoge(self):
        code = utils.clip_head("""
        def forward():
            x = 0
            for i in range(2):
                x = float(i) + 1
            return x
        """)
        pass

        tree = gast.ast_to_gast(ast.parse(code))
        node_id = generate_id_table(tree)
        node_type = generate_type_table(tree)
        pprint.pprint(node_id)
        pprint.pprint(node_type)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
