import ast, gast
import pprint
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward


class TestNum(unittest.TestCase):
    def test_num_bool(self):
        class Test():
            def forward(self, x, y):
                return x and y or True

        id2type = generate_id2type_from_forward(Test(), (True, False))

        self.assertEqual(str(id2type[1]), "Test -> bool -> bool -> bool")	# FunctionDef (line 2)
        self.assertEqual(str(id2type[9]), "bool")	# Return (line 3)
        self.assertEqual(str(id2type[10]), "bool")	# BoolOp (line 3)
        self.assertEqual(str(id2type[11]), "bool -> bool -> bool")	# Or
        self.assertEqual(str(id2type[12]), "bool")	# BoolOp (line 3)
        self.assertEqual(str(id2type[13]), "bool -> bool -> bool")	# And
        self.assertEqual(str(id2type[14]), "bool")	# Name (line 3)
        self.assertEqual(str(id2type[16]), "bool")	# Name (line 3)
        self.assertEqual(str(id2type[18]), "bool")	# NameConstant (line 3)


    def test_num_coersion(self):
        class Test():
            def forward(self, x):
                y = abs(x)
                x = x + 1.3
                return x

        id2type = generate_id2type_from_forward(Test(), (1,))

        self.assertEqual(str(id2type[1]), "Test -> int -> float")	# FunctionDef (line 2)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[8]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[10]), "int")	# Call (line 3)
        self.assertEqual(str(id2type[11]), "int -> int")	# Name (line 3)
        self.assertEqual(str(id2type[13]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[15]), "NoneType")	# Assign (line 4)
        self.assertEqual(str(id2type[16]), "float")	# Name (line 4)
        self.assertEqual(str(id2type[18]), "float")	# BinOp (line 4)
        self.assertEqual(str(id2type[19]), "int")	# Name (line 4)
        self.assertEqual(str(id2type[21]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[22]), "float")	# Num (line 4)
        self.assertEqual(str(id2type[23]), "float")	# Return (line 5)
        self.assertEqual(str(id2type[24]), "float")	# Name (line 5)


    def test_num_coersion_if(self):
        class Test():
            def forward(self):
                a = 1
                b = a
                if True:
                    b = b + 1.0
                return a

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> int")	# FunctionDef (line 2)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[6]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[8]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[9]), "NoneType")	# Assign (line 4)
        self.assertEqual(str(id2type[10]), "float")	# Name (line 4)
        self.assertEqual(str(id2type[12]), "float")	# Name (line 4)
        self.assertEqual(str(id2type[14]), "NoneType")	# If (line 5)
        self.assertEqual(str(id2type[15]), "bool")	# NameConstant (line 5)
        self.assertEqual(str(id2type[16]), "NoneType")	# Assign (line 6)
        self.assertEqual(str(id2type[17]), "float")	# Name (line 6)
        self.assertEqual(str(id2type[19]), "float")	# BinOp (line 6)
        self.assertEqual(str(id2type[20]), "int")	# Name (line 6)
        self.assertEqual(str(id2type[22]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[23]), "float")	# Num (line 6)
        self.assertEqual(str(id2type[24]), "int")	# Return (line 7)
        self.assertEqual(str(id2type[25]), "int")	# Name (line 7)


    def test_num_coersion_if_else(self):
        class Test():
            def forward(self, x):
                if True:
                    x += 3
                else:
                    x += 10.0
                return x

        id2type = generate_id2type_from_forward(Test(), (0,))  # int

        self.assertEqual(str(id2type[1]), "Test -> int -> float")	# FunctionDef (line 2)
        self.assertEqual(str(id2type[7]), "NoneType")	# If (line 3)
        self.assertEqual(str(id2type[8]), "bool")	# NameConstant (line 3)
        self.assertEqual(str(id2type[9]), "NoneType")	# AugAssign (line 4)
        self.assertEqual(str(id2type[10]), "float")	# Name (line 4)
        self.assertEqual(str(id2type[12]), "int -> int -> float")	# Add
        self.assertEqual(str(id2type[13]), "int")	# Num (line 4)
        self.assertEqual(str(id2type[14]), "NoneType")	# AugAssign (line 6)
        self.assertEqual(str(id2type[15]), "float")	# Name (line 6)
        self.assertEqual(str(id2type[17]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[18]), "float")	# Num (line 6)
        self.assertEqual(str(id2type[19]), "float")	# Return (line 7)
        self.assertEqual(str(id2type[20]), "float")	# Name (line 7)


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

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> int list list")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "[int, int, int]")	# Name (line 2)
        self.assertEqual(str(id2type[8]), "[int, int, int]")	# List (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[14]), "[]")	# Name (line 3)
        self.assertEqual(str(id2type[16]), "[]")	# List (line 3)
        self.assertEqual(str(id2type[18]), "NoneType")	# For (line 4)
        self.assertEqual(str(id2type[19]), "int")	# Name (line 4)
        self.assertEqual(str(id2type[21]), "int list")	# Call (line 4)
        self.assertEqual(str(id2type[22]), "int -> int list")	# Name (line 4)
        self.assertEqual(str(id2type[24]), "int")	# Num (line 4)
        self.assertEqual(str(id2type[25]), "NoneType")	# Expr (line 5)
        self.assertEqual(str(id2type[26]), "NoneType")	# Call (line 5)
        self.assertEqual(str(id2type[27]), "int list -> NoneType")	# Attribute (line 5)
        self.assertEqual(str(id2type[28]), "int list list")	# Name (line 5)
        self.assertEqual(str(id2type[31]), "int list")	# Subscript (line 5)
        self.assertEqual(str(id2type[32]), "int list")	# Name (line 5)
        self.assertEqual(str(id2type[35]), "int")	# Name (line 5)
        self.assertEqual(str(id2type[38]), "int list list")	# Return (line 6)
        self.assertEqual(str(id2type[39]), "int list list")	# Name (line 6)


    def test_tuple_coersion(self):
        class Test():
            def forward(self):
                x = (1, 2, 3)
                for i in range(1, 3):
                    o = x[i]
                return o

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> int")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "(int, int, int)")	# Name (line 2)
        self.assertEqual(str(id2type[8]), "(int, int, int)")	# Tuple (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# For (line 3)
        self.assertEqual(str(id2type[14]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[16]), "int list")	# Call (line 3)
        self.assertEqual(str(id2type[17]), "int -> int -> int list")	# Name (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[20]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[21]), "NoneType")	# Assign (line 4)
        self.assertEqual(str(id2type[22]), "int")	# Name (line 4)
        self.assertEqual(str(id2type[24]), "int")	# Subscript (line 4)
        self.assertEqual(str(id2type[25]), "int tuple")	# Name (line 4)
        self.assertEqual(str(id2type[28]), "int")	# Name (line 4)
        self.assertEqual(str(id2type[31]), "int")	# Return (line 5)
        self.assertEqual(str(id2type[32]), "int")	# Name (line 5)


    def test_tuple_coersion_2(self):
        class Test():
            def forward(self):
                x = (1, 2, 3)
                x += (4, 5, 6)
                return x

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> int tuple")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "(int, int, int)")	# Name (line 2)
        self.assertEqual(str(id2type[8]), "(int, int, int)")	# Tuple (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# AugAssign (line 3)
        self.assertEqual(str(id2type[14]), "int tuple")	# Name (line 3)
        self.assertEqual(str(id2type[16]), "int tuple -> int tuple -> int tuple")	# Add
        self.assertEqual(str(id2type[17]), "int tuple")	# Tuple (line 3)
        self.assertEqual(str(id2type[18]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[20]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[22]), "int tuple")	# Return (line 4)
        self.assertEqual(str(id2type[23]), "int tuple")	# Name (line 4)


    def test_tuple_assign(self):
        class Test():
            def forward(self):
                x, y = 1, 2.0
                return x + y

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> float")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "(int, float)")	# Tuple (line 2)
        self.assertEqual(str(id2type[7]), "int")	# Name (line 2)
        self.assertEqual(str(id2type[9]), "float")	# Name (line 2)
        self.assertEqual(str(id2type[12]), "(int, float)")	# Tuple (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[14]), "float")	# Num (line 2)
        self.assertEqual(str(id2type[16]), "float")	# Return (line 3)
        self.assertEqual(str(id2type[17]), "float")	# BinOp (line 3)
        self.assertEqual(str(id2type[18]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[20]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[21]), "float")	# Name (line 3)


    def test_list_slice(self):
        class Test():
            def forward(self):
                x = [0, 1, 2, 3]
                return x[1:2]

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> int list")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "[int, int, int, int]")	# Name (line 2)
        self.assertEqual(str(id2type[8]), "[int, int, int, int]")	# List (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[12]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[14]), "int list")	# Return (line 3)
        self.assertEqual(str(id2type[15]), "int list")	# Subscript (line 3)
        self.assertEqual(str(id2type[16]), "int list")	# Name (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[20]), "int")	# Num (line 3)


    def test_list_of_tuple(self):
        class Test():
            def forward(self, v):
                for x, y in [(1, 2.0), (2, 3.0)]:
                    v += x + y
                return v

        id2type = generate_id2type_from_forward(Test(), (0,))

        self.assertEqual(str(id2type[1]), "Test -> int -> float")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# For (line 2)
        self.assertEqual(str(id2type[8]), "(int, float)")	# Tuple (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Name (line 2)
        self.assertEqual(str(id2type[11]), "float")	# Name (line 2)
        self.assertEqual(str(id2type[14]), "(int, float) list")	# List (line 2)
        self.assertEqual(str(id2type[15]), "(int, float)")	# Tuple (line 2)
        self.assertEqual(str(id2type[16]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[17]), "float")	# Num (line 2)
        self.assertEqual(str(id2type[19]), "(int, float)")	# Tuple (line 2)
        self.assertEqual(str(id2type[20]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[21]), "float")	# Num (line 2)
        self.assertEqual(str(id2type[24]), "NoneType")	# AugAssign (line 3)
        self.assertEqual(str(id2type[25]), "float")	# Name (line 3)
        self.assertEqual(str(id2type[27]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[28]), "float")	# BinOp (line 3)
        self.assertEqual(str(id2type[29]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[31]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[32]), "float")	# Name (line 3)
        self.assertEqual(str(id2type[34]), "float")	# Return (line 4)
        self.assertEqual(str(id2type[35]), "float")	# Name (line 4)


# ==============================================================================

class TestOtherDataTypes(unittest.TestCase):
    def test_string(self):
        class Test():
            def forward(self):
                v = "foobar"
                for x in ["a", "b"]:
                    v += x
                return v

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> string")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "string")	# Name (line 2)
        self.assertEqual(str(id2type[8]), "string")	# Str (line 2)
        self.assertEqual(str(id2type[9]), "NoneType")	# For (line 3)
        self.assertEqual(str(id2type[10]), "string")	# Name (line 3)
        self.assertEqual(str(id2type[12]), "string list")	# List (line 3)
        self.assertEqual(str(id2type[13]), "string")	# Str (line 3)
        self.assertEqual(str(id2type[14]), "string")	# Str (line 3)
        self.assertEqual(str(id2type[16]), "NoneType")	# AugAssign (line 4)
        self.assertEqual(str(id2type[17]), "string")	# Name (line 4)
        self.assertEqual(str(id2type[19]), "string -> string -> string")	# Add
        self.assertEqual(str(id2type[20]), "string")	# Name (line 4)
        self.assertEqual(str(id2type[22]), "string")	# Return (line 5)
        self.assertEqual(str(id2type[23]), "string")	# Name (line 5)


    def test_dict_basic(self):
        class Test():
            def forward(self):
                x = {"one": 1, "two": 2}
                return x["one"]

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> int")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "{string : int}")	# Name (line 2)
        self.assertEqual(str(id2type[8]), "{string : int}")	# Dict (line 2)
        self.assertEqual(str(id2type[9]), "string")	# Str (line 2)
        self.assertEqual(str(id2type[10]), "string")	# Str (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[12]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Return (line 3)
        self.assertEqual(str(id2type[14]), "int")	# Subscript (line 3)
        self.assertEqual(str(id2type[15]), "{string : int}")	# Name (line 3)
        self.assertEqual(str(id2type[18]), "string")	# Str (line 3)


# ==============================================================================

class TestControl(unittest.TestCase):
    def test_for_simple(self):
        class Test():
            def forward(self, x):
                for i in range(2):
                    x = float(i) + 1
                return x

        id2type = generate_id2type_from_forward(Test(), (0,))

        self.assertEqual(str(id2type[1]), "Test -> int -> float")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# For (line 2)
        self.assertEqual(str(id2type[8]), "int")	# Name (line 2)
        self.assertEqual(str(id2type[10]), "int list")	# Call (line 2)
        self.assertEqual(str(id2type[11]), "int -> int list")	# Name (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[14]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[15]), "float")	# Name (line 3)
        self.assertEqual(str(id2type[17]), "float")	# BinOp (line 3)
        self.assertEqual(str(id2type[18]), "float")	# Call (line 3)
        self.assertEqual(str(id2type[19]), "int -> float")	# Name (line 3)
        self.assertEqual(str(id2type[21]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[23]), "float -> int -> float")	# Add
        self.assertEqual(str(id2type[24]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[25]), "float")	# Return (line 4)
        self.assertEqual(str(id2type[26]), "float")	# Name (line 4)


    ### template
    # def test_(self):
    #     code = utils.clip_head("""
    #     """)
    #     tree = gast.ast_to_gast(ast.parse(code))
    #     id2type = generate_id2type(tree)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
