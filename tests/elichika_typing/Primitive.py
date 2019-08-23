import ast, gast
import pprint
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward


import numpy as np
test_var = 42


class TestNum(unittest.TestCase):
    def test_num_bool(self):
        class Test():
            def forward(self, x, y):
                return x and y or True

        node_type = generate_id2type_from_forward(Test(), (True, False))

        self.assertEqual(str(node_type[1]), "bool -> bool -> bool")	# FunctionDef (line 2)
        self.assertEqual(str(node_type[9]), "bool")	# Return (line 3)
        self.assertEqual(str(node_type[10]), "bool")	# BoolOp (line 3)
        self.assertEqual(str(node_type[11]), "bool -> bool -> bool")	# Or
        self.assertEqual(str(node_type[12]), "bool")	# BoolOp (line 3)
        self.assertEqual(str(node_type[13]), "bool -> bool -> bool")	# And
        self.assertEqual(str(node_type[14]), "bool")	# Name (line 3)
        self.assertEqual(str(node_type[16]), "bool")	# Name (line 3)
        self.assertEqual(str(node_type[18]), "bool")	# NameConstant (line 3)


    def test_num_coersion(self):
        class Test():
            def forward(self, x):
                y = abs(x)
                x = x + 1.3
                return x

        node_type = generate_id2type_from_forward(Test(), (1,))

        self.assertEqual(str(node_type[1]), "int -> float")	# FunctionDef (line 2)
        self.assertEqual(str(node_type[7]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(node_type[8]), "int")	# Name (line 3)
        self.assertEqual(str(node_type[10]), "int")	# Call (line 3)
        self.assertEqual(str(node_type[11]), "int -> int")	# Name (line 3)
        self.assertEqual(str(node_type[13]), "int")	# Name (line 3)
        self.assertEqual(str(node_type[15]), "NoneType")	# Assign (line 4)
        self.assertEqual(str(node_type[16]), "float")	# Name (line 4)
        self.assertEqual(str(node_type[18]), "float")	# BinOp (line 4)
        self.assertEqual(str(node_type[19]), "int")	# Name (line 4)
        self.assertEqual(str(node_type[21]), "int -> float -> float")	# Add
        self.assertEqual(str(node_type[22]), "float")	# Num (line 4)
        self.assertEqual(str(node_type[23]), "float")	# Return (line 5)
        self.assertEqual(str(node_type[24]), "float")	# Name (line 5)


    def test_num_coersion_if(self):
        class Test():
            def forward(self):
                a = 1
                b = a
                if True:
                    b = b + 1.0
                return a

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> int")	# FunctionDef (line 2)
        self.assertEqual(str(node_type[5]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(node_type[6]), "int")	# Name (line 3)
        self.assertEqual(str(node_type[8]), "int")	# Num (line 3)
        self.assertEqual(str(node_type[9]), "NoneType")	# Assign (line 4)
        self.assertEqual(str(node_type[10]), "float")	# Name (line 4)
        self.assertEqual(str(node_type[12]), "float")	# Name (line 4)
        self.assertEqual(str(node_type[14]), "NoneType")	# If (line 5)
        self.assertEqual(str(node_type[15]), "bool")	# NameConstant (line 5)
        self.assertEqual(str(node_type[16]), "NoneType")	# Assign (line 6)
        self.assertEqual(str(node_type[17]), "float")	# Name (line 6)
        self.assertEqual(str(node_type[19]), "float")	# BinOp (line 6)
        self.assertEqual(str(node_type[20]), "int")	# Name (line 6)
        self.assertEqual(str(node_type[22]), "int -> float -> float")	# Add
        self.assertEqual(str(node_type[23]), "float")	# Num (line 6)
        self.assertEqual(str(node_type[24]), "int")	# Return (line 7)
        self.assertEqual(str(node_type[25]), "int")	# Name (line 7)


    def test_num_coersion_if_else(self):
        class Test():
            def forward(self, x):
                if True:
                    x += 3
                else:
                    x += 10.0
                return x

        node_type = generate_id2type_from_forward(Test(), (0,))  # int

        self.assertEqual(str(node_type[1]), "int -> float")	# FunctionDef (line 2)
        self.assertEqual(str(node_type[7]), "NoneType")	# If (line 3)
        self.assertEqual(str(node_type[8]), "bool")	# NameConstant (line 3)
        self.assertEqual(str(node_type[9]), "NoneType")	# AugAssign (line 4)
        self.assertEqual(str(node_type[10]), "float")	# Name (line 4)
        self.assertEqual(str(node_type[12]), "int -> int -> float")	# Add
        self.assertEqual(str(node_type[13]), "int")	# Num (line 4)
        self.assertEqual(str(node_type[14]), "NoneType")	# AugAssign (line 6)
        self.assertEqual(str(node_type[15]), "float")	# Name (line 6)
        self.assertEqual(str(node_type[17]), "int -> float -> float")	# Add
        self.assertEqual(str(node_type[18]), "float")	# Num (line 6)
        self.assertEqual(str(node_type[19]), "float")	# Return (line 7)
        self.assertEqual(str(node_type[20]), "float")	# Name (line 7)


# ==============================================================================

# TODO(momohatt): regenerate assertions for the following tests
class TestSequence(unittest.TestCase):
    def test_list(self):
        class Test():
            def forward(self):
                xs = [1, 2, 3]
                v = []
                for i in range(3):
                    v.append(xs[:i])
                return v

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> int list list")	# lineno: 2
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
        class Test():
            def forward(self):
                x = (1, 2, 3)
                for i in range(3):
                    o = x[i]
                return o

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> int")	# lineno: 2
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
        class Test():
            def forward(self):
                x = (1, 2, 3)
                x += (4, 5, 6)
                return x

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> int tuple")	# lineno: 2
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
        class Test():
            def forward(self):
                x, y = 1, 2.0
                return x + y

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> float")	# lineno: 2
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
        class Test():
            def forward(self):
                x = [0, 1, 2, 3]
                return x[1:2]

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> int list")	# lineno: 2
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
        class Test():
            def forward(self):
                v = 0
                for x, y in [(1, 2.0), (2, 3.0)]:
                    v += x + y
                return v

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> float")	# lineno: 2
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
        class Test():
            def forward(self):
                v = "foobar"
                for x in ["a", "b"]:
                    v += x
                return v

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> string")	# lineno: 2
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
        class Test():
            def forward(self):
                x = {"one": 1, "two": 2}
                return x["one"]

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> int")	# lineno: 2
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
        class Test():
            def forward(self):
                x = 0
                for i in range(2):
                    x = float(i) + 1
                return x

        node_type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(node_type[1]), "(no argument) -> float")	# lineno: 2
        self.assertEqual(str(node_type[5]), "NoneType")	# lineno: 3
        self.assertEqual(str(node_type[6]), "int")	# lineno: 3
        self.assertEqual(str(node_type[8]), "int")	# lineno: 3
        self.assertEqual(str(node_type[9]), "NoneType")	# lineno: 4
        self.assertEqual(str(node_type[10]), "int")	# lineno: 4
        self.assertEqual(str(node_type[12]), "int list")	# lineno: 4
        self.assertEqual(str(node_type[13]), "int -> int list")	# lineno: 4
        self.assertEqual(str(node_type[15]), "int")	# lineno: 4
        self.assertEqual(str(node_type[16]), "NoneType")	# lineno: 5
        self.assertEqual(str(node_type[17]), "float")	# lineno: 5
        self.assertEqual(str(node_type[19]), "float")	# lineno: 5
        self.assertEqual(str(node_type[20]), "float")	# lineno: 5
        self.assertEqual(str(node_type[21]), "int -> float")	# lineno: 5
        self.assertEqual(str(node_type[23]), "int")	# lineno: 5
        self.assertEqual(str(node_type[25]), "float -> int -> float")
        self.assertEqual(str(node_type[26]), "int")	# lineno: 5
        self.assertEqual(str(node_type[27]), "float")	# lineno: 6
        self.assertEqual(str(node_type[28]), "float")	# lineno: 6


    ### template
    # def test_(self):
    #     code = utils.clip_head("""
    #     """)
    #     tree = gast.ast_to_gast(ast.parse(code))
    #     node_type = generate_id2type(tree)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
