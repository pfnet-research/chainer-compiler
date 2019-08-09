import ast, gast
import pprint
import unittest

from chainer_compiler.elichika.parser import typing
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import generate_type_table


class TestPrimitive(unittest.TestCase):
    def test_hoge(self):
        code = utils.clip_head("""
        def forward():
            x = 0
            for i in range(2):
                x = float(i) + 1
            return x
        """)

        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)  # id -> type

        assert str(node_type[6]) == "int"
        assert str(node_type[4]) == "int"
        assert str(node_type[3]) == "NoneType"
        assert str(node_type[13]) == "int"
        assert str(node_type[11]) == "int -> int list"
        assert str(node_type[10]) == "int list"
        assert str(node_type[8]) == "int"
        assert str(node_type[21]) == "int"
        assert str(node_type[19]) == "int -> float"
        assert str(node_type[18]) == "float"
        assert str(node_type[24]) == "int"
        assert str(node_type[23]) == "float -> int -> float"
        assert str(node_type[17]) == "float"
        assert str(node_type[15]) == "float"
        assert str(node_type[14]) == "NoneType"
        assert str(node_type[7]) == "NoneType"
        assert str(node_type[26]) == "float"
        assert str(node_type[25]) == "float"
        assert str(node_type[1]) == "float"


def main():
    unittest.main()


if __name__ == '__main__':
    main()
