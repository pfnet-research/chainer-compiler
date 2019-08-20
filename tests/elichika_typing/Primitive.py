import ast, gast
import pprint
import unittest

from chainer_compiler.elichika.parser import typing
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.testtools import generate_type_table


class TestNum(unittest.TestCase):
    def test_num_bool(self):
        code = utils.clip_head("""
        def forward(self, x, y):
            return x and y or True
        """)

        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "bool")	# lineno: 2
        self.assertEqual(str(node_type[9]), "bool")	# lineno: 3
        self.assertEqual(str(node_type[10]), "bool")	# lineno: 3
        self.assertEqual(str(node_type[11]), "bool -> bool -> bool")
        self.assertEqual(str(node_type[12]), "bool")	# lineno: 3
        self.assertEqual(str(node_type[13]), "bool -> bool -> bool")
        self.assertEqual(str(node_type[14]), "bool")	# lineno: 3
        self.assertEqual(str(node_type[16]), "bool")	# lineno: 3
        self.assertEqual(str(node_type[18]), "bool")	# lineno: 3


    def test_num_coersion(self):
        code = utils.clip_head("""
        def forward(self):
            x = 0
            y = abs(x)
            x = x + 1.3
            return x
        """)

        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "float")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "int")	# lineno: 3
        self.assertEqual(str(node_type[8]), "int")	# lineno: 3
        self.assertEqual(str(node_type[9]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[10]), "int")	# lineno: 4
        self.assertEqual(str(node_type[12]), "int")	# lineno: 4
        self.assertEqual(str(node_type[13]), "int -> int")	# lineno: 4
        self.assertEqual(str(node_type[15]), "int")	# lineno: 4
        self.assertEqual(str(node_type[17]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[18]), "float")	# lineno: 5
        self.assertEqual(str(node_type[20]), "float")	# lineno: 5
        self.assertEqual(str(node_type[21]), "int")	# lineno: 5
        self.assertEqual(str(node_type[23]), "int -> float -> float")
        self.assertEqual(str(node_type[24]), "float")	# lineno: 5
        self.assertEqual(str(node_type[25]), "float")	# lineno: 6
        self.assertEqual(str(node_type[26]), "float")	# lineno: 6


    def test_num_coersion_if(self):
        code = utils.clip_head("""
        def forward(self):
            a = 1
            b = a
            if True:
                b = b + 1.0
            return a
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "int")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "int")	# lineno: 3
        self.assertEqual(str(node_type[8]), "int")	# lineno: 3
        self.assertEqual(str(node_type[9]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[10]), "float")	# lineno: 4
        self.assertEqual(str(node_type[12]), "float")	# lineno: 4
        self.assertEqual(str(node_type[14]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[15]), "bool")	# lineno: 5
        self.assertEqual(str(node_type[16]), "NoneType")	# lineno: 6
        self.assertEqual(str(node_type[17]), "float")	# lineno: 6
        self.assertEqual(str(node_type[19]), "float")	# lineno: 6
        self.assertEqual(str(node_type[20]), "int")	# lineno: 6
        self.assertEqual(str(node_type[22]), "int -> float -> float")
        self.assertEqual(str(node_type[23]), "float")	# lineno: 6
        self.assertEqual(str(node_type[24]), "int")	# lineno: 7
        self.assertEqual(str(node_type[25]), "int")	# lineno: 7


    def test_num_coersion_if_else(self):
        code = utils.clip_head("""
        def forward(self):
            x = 0
            if True:
                x += 3
            else:
                x += 10.0
            return x
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "float")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "int")	# lineno: 3
        self.assertEqual(str(node_type[8]), "int")	# lineno: 3
        self.assertEqual(str(node_type[9]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[10]), "bool")	# lineno: 4
        self.assertEqual(str(node_type[11]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[12]), "float")	# lineno: 5
        self.assertEqual(str(node_type[14]), "int -> int -> float")
        self.assertEqual(str(node_type[15]), "int")	# lineno: 5
        self.assertEqual(str(node_type[16]), "NoneType")	# lineno: 7
        self.assertEqual(str(node_type[17]), "float")	# lineno: 7
        self.assertEqual(str(node_type[19]), "int -> float -> float")
        self.assertEqual(str(node_type[20]), "float")	# lineno: 7
        self.assertEqual(str(node_type[21]), "float")	# lineno: 8
        self.assertEqual(str(node_type[22]), "float")	# lineno: 8


# ==============================================================================

class TestSequence(unittest.TestCase):
    def test_list(self):
        code = utils.clip_head("""
        def forward(self):
            xs = [1, 2, 3]
            v = []
            for i in range(3):
                v.append(xs[:i])
            return v
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "int list list")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "[int, int, int]")	# lineno: 3
        self.assertEqual(str(node_type[8]), "[int, int, int]")	# lineno: 3
        self.assertEqual(str(node_type[9]), "int")	# lineno: 3
        self.assertEqual(str(node_type[10]), "int")	# lineno: 3
        self.assertEqual(str(node_type[11]), "int")	# lineno: 3
        self.assertEqual(str(node_type[13]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[14]), "[]")	# lineno: 4
        self.assertEqual(str(node_type[16]), "[]")	# lineno: 4
        self.assertEqual(str(node_type[18]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[19]), "int")	# lineno: 5
        self.assertEqual(str(node_type[21]), "int list")	# lineno: 5
        self.assertEqual(str(node_type[22]), "int -> int list")	# lineno: 5
        self.assertEqual(str(node_type[24]), "int")	# lineno: 5
        self.assertEqual(str(node_type[25]), "NoneType")	# lineno: 6
        self.assertEqual(str(node_type[26]), "NoneType")	# lineno: 6
        self.assertEqual(str(node_type[27]), "int list -> NoneType")	# lineno: 6
        self.assertEqual(str(node_type[28]), "int list list")	# lineno: 6
        self.assertEqual(str(node_type[31]), "int list")	# lineno: 6
        self.assertEqual(str(node_type[32]), "int list")	# lineno: 6
        self.assertEqual(str(node_type[35]), "int")	# lineno: 6
        self.assertEqual(str(node_type[38]), "int list list")	# lineno: 7
        self.assertEqual(str(node_type[39]), "int list list")	# lineno: 7


    def test_tuple_coersion(self):
        code = utils.clip_head("""
        def forward(self):
            x = (1, 2, 3)
            for i in range(3):
                o = x[i]
            return o
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "int")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "(int, int, int)")	# lineno: 3
        self.assertEqual(str(node_type[8]), "(int, int, int)")	# lineno: 3
        self.assertEqual(str(node_type[9]), "int")	# lineno: 3
        self.assertEqual(str(node_type[10]), "int")	# lineno: 3
        self.assertEqual(str(node_type[11]), "int")	# lineno: 3
        self.assertEqual(str(node_type[13]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[14]), "int")	# lineno: 4
        self.assertEqual(str(node_type[16]), "int list")	# lineno: 4
        self.assertEqual(str(node_type[17]), "int -> int list")	# lineno: 4
        self.assertEqual(str(node_type[19]), "int")	# lineno: 4
        self.assertEqual(str(node_type[20]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[21]), "int")	# lineno: 5
        self.assertEqual(str(node_type[23]), "int")	# lineno: 5
        self.assertEqual(str(node_type[24]), "int tuple")	# lineno: 5
        self.assertEqual(str(node_type[27]), "int")	# lineno: 5
        self.assertEqual(str(node_type[30]), "int")	# lineno: 6
        self.assertEqual(str(node_type[31]), "int")	# lineno: 6


    def test_tuple_coersion_2(self):
        code = utils.clip_head("""
        def forward(self):
            x = (1, 2, 3)
            x += (4, 5, 6)
            return x
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "int tuple")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "(int, int, int)")	# lineno: 3
        self.assertEqual(str(node_type[8]), "(int, int, int)")	# lineno: 3
        self.assertEqual(str(node_type[9]), "int")	# lineno: 3
        self.assertEqual(str(node_type[10]), "int")	# lineno: 3
        self.assertEqual(str(node_type[11]), "int")	# lineno: 3
        self.assertEqual(str(node_type[13]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[14]), "int tuple")	# lineno: 4
        self.assertEqual(str(node_type[16]), "int tuple -> int tuple -> int tuple")
        self.assertEqual(str(node_type[17]), "int tuple")	# lineno: 4
        self.assertEqual(str(node_type[18]), "int")	# lineno: 4
        self.assertEqual(str(node_type[19]), "int")	# lineno: 4
        self.assertEqual(str(node_type[20]), "int")	# lineno: 4
        self.assertEqual(str(node_type[22]), "int tuple")	# lineno: 5
        self.assertEqual(str(node_type[23]), "int tuple")	# lineno: 5


    def test_tuple_assign(self):
        code = utils.clip_head("""
        def forward(self):
            x, y = 1, 2.0
            return x + y
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "float")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "(int, float)")	# lineno: 3
        self.assertEqual(str(node_type[7]), "int")	# lineno: 3
        self.assertEqual(str(node_type[9]), "float")	# lineno: 3
        self.assertEqual(str(node_type[12]), "(int, float)")	# lineno: 3
        self.assertEqual(str(node_type[13]), "int")	# lineno: 3
        self.assertEqual(str(node_type[14]), "float")	# lineno: 3
        self.assertEqual(str(node_type[16]), "float")	# lineno: 4
        self.assertEqual(str(node_type[17]), "float")	# lineno: 4
        self.assertEqual(str(node_type[18]), "int")	# lineno: 4
        self.assertEqual(str(node_type[20]), "int -> float -> float")
        self.assertEqual(str(node_type[21]), "float")	# lineno: 4


    def test_list_slice(self):
        code = utils.clip_head("""
        def forward(self):
            x = [0, 1, 2, 3]
            return x[1:2]
        """)

        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "int list")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "[int, int, int, int]")	# lineno: 3
        self.assertEqual(str(node_type[8]), "[int, int, int, int]")	# lineno: 3
        self.assertEqual(str(node_type[9]), "int")	# lineno: 3
        self.assertEqual(str(node_type[10]), "int")	# lineno: 3
        self.assertEqual(str(node_type[11]), "int")	# lineno: 3
        self.assertEqual(str(node_type[12]), "int")	# lineno: 3
        self.assertEqual(str(node_type[14]), "int list")	# lineno: 4
        self.assertEqual(str(node_type[15]), "int list")	# lineno: 4
        self.assertEqual(str(node_type[16]), "int list")	# lineno: 4
        self.assertEqual(str(node_type[19]), "int")	# lineno: 4
        self.assertEqual(str(node_type[20]), "int")	# lineno: 4


    def test_list_of_tuple(self):
        code = utils.clip_head("""
        def forward(self):
            v = 0
            for x, y in [(1, 2.0), (2, 3.0)]:
                v += x + y
            return v
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "float")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "int")	# lineno: 3
        self.assertEqual(str(node_type[8]), "int")	# lineno: 3
        self.assertEqual(str(node_type[9]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[10]), "(int, float)")	# lineno: 4
        self.assertEqual(str(node_type[11]), "int")	# lineno: 4
        self.assertEqual(str(node_type[13]), "float")	# lineno: 4
        self.assertEqual(str(node_type[16]), "(int, float) list")	# lineno: 4
        self.assertEqual(str(node_type[17]), "(int, float)")	# lineno: 4
        self.assertEqual(str(node_type[18]), "int")	# lineno: 4
        self.assertEqual(str(node_type[19]), "float")	# lineno: 4
        self.assertEqual(str(node_type[21]), "(int, float)")	# lineno: 4
        self.assertEqual(str(node_type[22]), "int")	# lineno: 4
        self.assertEqual(str(node_type[23]), "float")	# lineno: 4
        self.assertEqual(str(node_type[26]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[27]), "float")	# lineno: 5
        self.assertEqual(str(node_type[29]), "int -> float -> float")
        self.assertEqual(str(node_type[30]), "float")	# lineno: 5
        self.assertEqual(str(node_type[31]), "int")	# lineno: 5
        self.assertEqual(str(node_type[33]), "int -> float -> float")
        self.assertEqual(str(node_type[34]), "float")	# lineno: 5
        self.assertEqual(str(node_type[36]), "float")	# lineno: 6
        self.assertEqual(str(node_type[37]), "float")	# lineno: 6


# ==============================================================================

class TestOtherDataTypes(unittest.TestCase):
    def test_string(self):
        code = utils.clip_head("""
        def forward(self):
            v = "foobar"
            for x in ["a", "b"]:
                v += x
            return v
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "string")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "string")	# lineno: 3
        self.assertEqual(str(node_type[8]), "string")	# lineno: 3
        self.assertEqual(str(node_type[9]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[10]), "string")	# lineno: 4
        self.assertEqual(str(node_type[12]), "string list")	# lineno: 4
        self.assertEqual(str(node_type[13]), "string")	# lineno: 4
        self.assertEqual(str(node_type[14]), "string")	# lineno: 4
        self.assertEqual(str(node_type[16]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[17]), "string")	# lineno: 5
        self.assertEqual(str(node_type[19]), "string -> string -> string")
        self.assertEqual(str(node_type[20]), "string")	# lineno: 5
        self.assertEqual(str(node_type[22]), "string")	# lineno: 6
        self.assertEqual(str(node_type[23]), "string")	# lineno: 6


    def test_dict_basic(self):
        code = utils.clip_head("""
        def forward(self):
            x = {"one": 1, "two": 2}
            return x["one"]
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "int")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "{string : int}")	# lineno: 3
        self.assertEqual(str(node_type[8]), "{string : int}")	# lineno: 3
        self.assertEqual(str(node_type[9]), "string")	# lineno: 3
        self.assertEqual(str(node_type[10]), "string")	# lineno: 3
        self.assertEqual(str(node_type[11]), "int")	# lineno: 3
        self.assertEqual(str(node_type[12]), "int")	# lineno: 3
        self.assertEqual(str(node_type[13]), "int")	# lineno: 4
        self.assertEqual(str(node_type[14]), "int")	# lineno: 4
        self.assertEqual(str(node_type[15]), "{string : int}")	# lineno: 4
        self.assertEqual(str(node_type[18]), "string")	# lineno: 4


# ==============================================================================

class TestControl(unittest.TestCase):
    def test_for_simple(self):
        code = utils.clip_head("""
        def forward():
            x = 0
            for i in range(2):
                x = float(i) + 1
            return x
        """)
        tree = gast.ast_to_gast(ast.parse(code))
        node_type = generate_type_table(tree)

        self.assertEqual(str(node_type[1]), "float")	# lineno: 2
        self.assertEqual(str(node_type[3]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[4]), "int")	# lineno: 3
        self.assertEqual(str(node_type[6]), "int")	# lineno: 3
        self.assertEqual(str(node_type[7]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[8]), "int")	# lineno: 4
        self.assertEqual(str(node_type[10]), "int list")	# lineno: 4
        self.assertEqual(str(node_type[11]), "int -> int list")	# lineno: 4
        self.assertEqual(str(node_type[13]), "int")	# lineno: 4
        self.assertEqual(str(node_type[14]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[15]), "float")	# lineno: 5
        self.assertEqual(str(node_type[17]), "float")	# lineno: 5
        self.assertEqual(str(node_type[18]), "float")	# lineno: 5
        self.assertEqual(str(node_type[19]), "int -> float")	# lineno: 5
        self.assertEqual(str(node_type[21]), "int")	# lineno: 5
        self.assertEqual(str(node_type[23]), "float -> int -> float")
        self.assertEqual(str(node_type[24]), "int")	# lineno: 5
        self.assertEqual(str(node_type[25]), "float")	# lineno: 6
        self.assertEqual(str(node_type[26]), "float")	# lineno: 6


    ### template
    # def test_(self):
    #     code = utils.clip_head("""
    #     """)
    #     tree = gast.ast_to_gast(ast.parse(code))
    #     node_type = generate_type_table(tree)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
