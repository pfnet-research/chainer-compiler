import ast, gast
import pprint
import unittest

from chainer_compiler.elichika.parser import typing
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import generate_type_table


class TestPrimitive(unittest.TestCase):
    def test_for(self):
        code = utils.clip_head("""
        def forward():
            x = 0
            for i in range(2):
                x = float(i) + 1
            return x
        """)

        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)  # id -> type

        assert str(node_type[1]) == "float"	# lineno: 2
        assert str(node_type[3]) == "NoneType"	# lineno: 3
        assert str(node_type[4]) == "int"	# lineno: 3
        assert str(node_type[6]) == "int"	# lineno: 3
        assert str(node_type[7]) == "NoneType"	# lineno: 4
        assert str(node_type[8]) == "int"	# lineno: 4
        assert str(node_type[10]) == "int list"	# lineno: 4
        assert str(node_type[11]) == "int -> int list"	# lineno: 4
        assert str(node_type[13]) == "int"	# lineno: 4
        assert str(node_type[14]) == "NoneType"	# lineno: 5
        assert str(node_type[15]) == "float"	# lineno: 5
        assert str(node_type[17]) == "float"	# lineno: 5
        assert str(node_type[18]) == "float"	# lineno: 5
        assert str(node_type[19]) == "int -> float"	# lineno: 5
        assert str(node_type[21]) == "int"	# lineno: 5
        assert str(node_type[23]) == "float -> int -> float"
        assert str(node_type[24]) == "int"	# lineno: 5
        assert str(node_type[25]) == "float"	# lineno: 6
        assert str(node_type[26]) == "float"	# lineno: 6


def main():
    unittest.main()


if __name__ == '__main__':
    main()
