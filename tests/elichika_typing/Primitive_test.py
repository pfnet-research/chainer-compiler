import ast, gast
import pprint
import pytest
import unittest

import chainer

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools


class TestNum(unittest.TestCase):
    # TODO(momohatt): regenerate test for updated comments

    def test_num_bool(self):
        class Test():
            def forward(self, x, y):
                return x and y or True

        id2type = generate_id2type_from_forward(Test(), (True, False))

        self.assertEqual(str(id2type[1]), "class Test -> bool -> bool -> bool")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[9]), "bool")	# Return (line 2)
        self.assertEqual(str(id2type[10]), "bool")	# BoolOp (line 2)
        self.assertEqual(str(id2type[11]), "bool -> bool -> bool")	# Or
        self.assertEqual(str(id2type[12]), "bool")	# BoolOp (line 2)
        self.assertEqual(str(id2type[13]), "bool -> bool -> bool")	# And
        self.assertEqual(str(id2type[14]), "bool")	# Name (line 2)
        self.assertEqual(str(id2type[16]), "bool")	# Name (line 2)
        self.assertEqual(str(id2type[18]), "bool")	# NameConstant (line 2)


    def test_num_coercion(self):
        class Test():
            def forward(self, x):
                y = abs(x)
                x = x + 1.3
                return x

        id2type = generate_id2type_from_forward(Test(), (1,))

        self.assertEqual(str(id2type[1]), "class Test -> int -> float")	# FunctionDef (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[8]), "int")	# Name (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Call (line 2)
        self.assertEqual(str(id2type[11]), "int -> int")	# Name (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Name (line 2)
        self.assertEqual(str(id2type[15]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[16]), "float")	# Name (line 3)
        self.assertEqual(str(id2type[18]), "float")	# BinOp (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Name (line 3)
        self.assertEqual(str(id2type[21]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[22]), "float")	# Num (line 3)
        self.assertEqual(str(id2type[23]), "float")	# Return (line 4)
        self.assertEqual(str(id2type[24]), "float")	# Name (line 4)


    def test_num_coercion_if(self):
        class Test():
            def forward(self):
                a = 1
                b = a
                if True:
                    b = b + 1.0
                return a

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "int")	# Name a (line 2)
        self.assertEqual(str(id2type[8]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[9]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[10]), "int")	# Name b (line 3)
        self.assertEqual(str(id2type[12]), "int")	# Name a (line 3)
        self.assertEqual(str(id2type[14]), "NoneType")	# If (line 4)
        self.assertEqual(str(id2type[15]), "bool")	# NameConstant (line 4)
        self.assertEqual(str(id2type[16]), "NoneType")	# Assign (line 5)
        self.assertEqual(str(id2type[17]), "float")	# Name b (line 5)
        self.assertEqual(str(id2type[19]), "float")	# BinOp (line 5)
        self.assertEqual(str(id2type[20]), "int")	# Name b (line 5)
        self.assertEqual(str(id2type[22]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[23]), "float")	# Num (line 5)
        self.assertEqual(str(id2type[24]), "int")	# Return (line 6)
        self.assertEqual(str(id2type[25]), "int")	# Name a (line 6)


    def test_num_coercion_if_else(self):
        class Test():
            def forward(self, x):
                if True:
                    x += 3
                else:
                    x += 10.0
                return x

        id2type = generate_id2type_from_forward(Test(), (0,))  # int

        self.assertEqual(str(id2type[1]), "class Test -> int -> float")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# If (line 2)
        self.assertEqual(str(id2type[8]), "bool")	# NameConstant (line 2)
        self.assertEqual(str(id2type[9]), "NoneType")	# AugAssign (line 3)
        self.assertEqual(str(id2type[10]), "int")	# Name x (line 3)
        self.assertEqual(str(id2type[12]), "int -> int -> int")	# Add
        self.assertEqual(str(id2type[13]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[14]), "NoneType")	# AugAssign (line 5)
        self.assertEqual(str(id2type[15]), "float")	# Name x (line 5)
        self.assertEqual(str(id2type[17]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[18]), "float")	# Num (line 5)
        self.assertEqual(str(id2type[19]), "float")	# Return (line 6)
        self.assertEqual(str(id2type[20]), "float")	# Name x (line 6)


# ==============================================================================

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

        self.assertEqual(str(id2type[1]), "class Test -> int list list")	# FunctionDef (line 1)
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
        self.assertEqual(str(id2type[26]), "NoneType")	# Call (line 5)
        self.assertEqual(str(id2type[27]), "int list -> NoneType")	# Attribute (line 5)
        self.assertEqual(str(id2type[28]), "int list list")	# Name (line 5)
        self.assertEqual(str(id2type[31]), "int list")	# Subscript (line 5)
        self.assertEqual(str(id2type[32]), "int list")	# Name (line 5)
        self.assertEqual(str(id2type[35]), "int")	# Name (line 5)
        self.assertEqual(str(id2type[38]), "int list list")	# Return (line 6)
        self.assertEqual(str(id2type[39]), "int list list")	# Name (line 6)


    def test_tuple_coercion(self):
        class Test():
            def forward(self):
                x = (1, 2, 3)
                for i in range(1, 3):
                    o = x[i]
                return o

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> int")	# FunctionDef (line 1)
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


    def test_tuple_coercion_2(self):
        class Test():
            def forward(self):
                x = (1, 2)
                x += (3,)
                return x

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> (int, int, int)")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "(int, int)")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "(int, int)")	# Tuple (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[12]), "NoneType")	# AugAssign (line 3)
        self.assertEqual(str(id2type[13]), "(int, int, int)")	# Name x (line 3)
        self.assertEqual(str(id2type[15]), "(int, int) -> (int,) -> (int, int, int)")	# Add
        self.assertEqual(str(id2type[16]), "(int,)")	# Tuple (line 3)
        self.assertEqual(str(id2type[17]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[19]), "(int, int, int)")	# Return (line 4)
        self.assertEqual(str(id2type[20]), "(int, int, int)")	# Name x (line 4)


    def test_tuple_assign(self):
        class Test():
            def forward(self):
                x, y = 1, 2.0
                return x + y

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> float")	# FunctionDef (line 1)
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

        self.assertEqual(str(id2type[1]), "class Test -> [int]")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign
        self.assertEqual(str(id2type[6]), "[int, int, int, int]")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "[int, int, int, int]")	# List [0, 1, 2, 3] (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num 0 (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[12]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[14]), "[int]")	# Return
        self.assertEqual(str(id2type[15]), "[int]")	# Subscript x[1:2:] (line 3)
        self.assertEqual(str(id2type[16]), "[int, int, int, int]")	# Name x (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[20]), "int")	# Num 2 (line 3)


    def test_list_of_tuple(self):
        class Test():
            def forward(self, v):
                for x, y in [(1, 2.0), (2, 3.0)]:
                    v += x + y
                return v

        id2type = generate_id2type_from_forward(Test(), (0,))

        self.assertEqual(str(id2type[1]), "class Test -> float -> float")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# For
        self.assertEqual(str(id2type[8]), "(int, float)")	# Tuple (x, y) (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[11]), "float")	# Name y (line 2)
        self.assertEqual(str(id2type[14]), "(int, float) list")	# List [(1, 2.0), (2, 3.0)] (line 2)
        self.assertEqual(str(id2type[15]), "(int, float)")	# Tuple (1, 2.0) (line 2)
        self.assertEqual(str(id2type[16]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[17]), "float")	# Num 2.0 (line 2)
        self.assertEqual(str(id2type[19]), "(int, float)")	# Tuple (2, 3.0) (line 2)
        self.assertEqual(str(id2type[20]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[21]), "float")	# Num 3.0 (line 2)
        self.assertEqual(str(id2type[24]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[25]), "float")	# Name v (line 3)
        self.assertEqual(str(id2type[27]), "float -> float -> float")	# Add
        self.assertEqual(str(id2type[28]), "float")	# BinOp x + y (line 3)
        self.assertEqual(str(id2type[29]), "int")	# Name x (line 3)
        self.assertEqual(str(id2type[31]), "int -> float -> float")	# Add
        self.assertEqual(str(id2type[32]), "float")	# Name y (line 3)
        self.assertEqual(str(id2type[34]), "float")	# Return
        self.assertEqual(str(id2type[35]), "float")	# Name v (line 4)


    def test_list_comprehension(self):
        class Test():
            def f(self, x):
                return x

            def forward(self, x):
                y = [self.f(i) for i in range(x)]
                return y

        id2type = generate_id2type_from_forward(Test(), (1,))

        self.assertEqual(str(id2type[1]), "class Test -> int -> int list")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[8]), "int list")	# Name y (line 2)
        self.assertEqual(str(id2type[10]), "int list")	# ListComp  (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Call self.f(i) (line 2)
        self.assertEqual(str(id2type[12]), "class Test -> int -> int")	# Attribute self.f (line 2)
        self.assertEqual(str(id2type[13]), "class Test")	# Name self (line 2)
        self.assertEqual(str(id2type[16]), "int")	# Name i (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Name i (line 2)
        self.assertEqual(str(id2type[21]), "int list")	# Call range(x) (line 2)
        self.assertEqual(str(id2type[22]), "int -> int list")	# Name range (line 2)
        self.assertEqual(str(id2type[24]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[26]), "int list")	# Return (line 3)
        self.assertEqual(str(id2type[27]), "int list")	# Name y (line 3)
        self.assertEqual(str(id2type[29]), "class Test -> int -> int")	# FunctionDef f (line 1)
        self.assertEqual(str(id2type[35]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[36]), "int")	# Name x (line 2)


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

        self.assertEqual(str(id2type[1]), "class Test -> string")	# FunctionDef (line 1)
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

        self.assertEqual(str(id2type[1]), "class Test -> int")	# FunctionDef (line 1)
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


    def test_optional(self):
        class Test():
            def forward(self):
                if False:
                    x = 1
                else:
                    x = None
                return x

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> optional(int)")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# If (line 2)
        self.assertEqual(str(id2type[6]), "bool")	# NameConstant (line 2)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[8]), "int")	# Name x (line 3)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[11]), "NoneType")	# Assign (line 5)
        self.assertEqual(str(id2type[12]), "NoneType")	# Name x (line 5)
        self.assertEqual(str(id2type[14]), "NoneType")	# NameConstant (line 5)
        self.assertEqual(str(id2type[15]), "optional(int)")	# Return (line 6)
        self.assertEqual(str(id2type[16]), "optional(int)")	# Name x (line 6)


# ==============================================================================

class TestControl(unittest.TestCase):
    def test_for_simple(self):
        class Test():
            def forward(self, x):
                for i in range(2):
                    x = float(i) + 1
                return x

        id2type = generate_id2type_from_forward(Test(), (0,))

        self.assertEqual(str(id2type[1]), "class Test -> float -> float")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# For (line 2)
        self.assertEqual(str(id2type[8]), "int")	# Name i (line 2)
        self.assertEqual(str(id2type[10]), "int list")	# Call (line 2)
        self.assertEqual(str(id2type[11]), "int -> int list")	# Name range (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[14]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[15]), "float")	# Name x (line 3)
        self.assertEqual(str(id2type[17]), "float")	# BinOp (line 3)
        self.assertEqual(str(id2type[18]), "float")	# Call (line 3)
        self.assertEqual(str(id2type[19]), "int -> float")	# Name float (line 3)
        self.assertEqual(str(id2type[21]), "int")	# Name i (line 3)
        self.assertEqual(str(id2type[23]), "float -> int -> float")	# Add
        self.assertEqual(str(id2type[24]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[25]), "float")	# Return (line 4)
        self.assertEqual(str(id2type[26]), "float")	# Name x (line 4)


    def test_for_optional(self):
        class Test():
            def f(self, x):
                if x is None:
                    x = 2
                return x

            def forward(self, x):
                for i in range(4):
                    x = self.f(x)
                return x

        id2type = generate_id2type_from_forward(Test(), (None,))

        self.assertEqual(str(id2type[1]), "class Test -> NoneType -> optional(int)")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# For
        self.assertEqual(str(id2type[8]), "int")	# Name i (line 2)
        self.assertEqual(str(id2type[10]), "int list")	# Call range(4) (line 2)
        self.assertEqual(str(id2type[11]), "int -> int list")	# Name range (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Num 4 (line 2)
        self.assertEqual(str(id2type[14]), "NoneType")	# Assign
        self.assertEqual(str(id2type[15]), "int")	# Name x (line 3)               (XXX: This is x on lhs)
        self.assertEqual(str(id2type[17]), "int")	# Call self.f(x) (line 3)
        self.assertEqual(str(id2type[18]), "class Test -> optional(int) -> int")	# Attribute self.f (line 3)
        self.assertEqual(str(id2type[19]), "class Test")	# Name self (line 3)
        self.assertEqual(str(id2type[22]), "optional(int)")	# Name x (line 3)
        self.assertEqual(str(id2type[24]), "optional(int)")	# Return
        self.assertEqual(str(id2type[25]), "optional(int)")	# Name x (line 4)
        self.assertEqual(str(id2type[27]), "class Test -> optional(int) -> int")	# FunctionDef f (line 1)
        self.assertEqual(str(id2type[33]), "NoneType")	# If
        self.assertEqual(str(id2type[34]), "bool")	# Compare  (line 2)
        self.assertEqual(str(id2type[35]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[39]), "NoneType")	# Assign
        self.assertEqual(str(id2type[40]), "int")	# Name x (line 3)
        self.assertEqual(str(id2type[42]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[43]), "int")	# Return
        self.assertEqual(str(id2type[44]), "int")	# Name x (line 4)


    def test_scope_if_else(self):
        class Test():
            def forward(self):
                if False:
                    o = 1
                else:
                    p = 2
                return o

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# If
        self.assertEqual(str(id2type[6]), "bool")	# NameConstant  (line 2)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[8]), "int")	# Name o (line 3)
        self.assertEqual(str(id2type[10]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[11]), "NoneType")	# Assign
        self.assertEqual(str(id2type[12]), "int")	# Name p (line 5)
        self.assertEqual(str(id2type[14]), "int")	# Num 2 (line 5)
        self.assertEqual(str(id2type[15]), "int")	# Return
        self.assertEqual(str(id2type[16]), "int")	# Name o (line 6)


class TestAssign(unittest.TestCase):
    def test_immutable_augassign(self):
        class Test():
            def forward(self):
                x = (1, 2, 3)
                y = x
                x += (4,)
                return y

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> (int, int, int)")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "(int, int, int)")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "(int, int, int)")	# Tuple (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[14]), "(int, int, int)")	# Name y (line 3)
        self.assertEqual(str(id2type[16]), "(int, int, int)")	# Name x (line 3)
        self.assertEqual(str(id2type[18]), "NoneType")	# AugAssign (line 4)
        self.assertEqual(str(id2type[19]), "(int, int, int, int)")	# Name x (line 4)
        self.assertEqual(str(id2type[21]), "(int, int, int) -> (int,) -> (int, int, int, int)")	# Add
        self.assertEqual(str(id2type[22]), "(int,)")	# Tuple (line 4)
        self.assertEqual(str(id2type[23]), "int")	# Num (line 4)
        self.assertEqual(str(id2type[25]), "(int, int, int)")	# Return (line 5)
        self.assertEqual(str(id2type[26]), "(int, int, int)")	# Name y (line 5)


    def test_mutable_augassign(self):
        class Test():
            def forward(self):
                x = [1, 2, 3]
                y = x
                x += [4]
                return y

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> int list")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "[int, int, int]")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "[int, int, int]")	# List (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[10]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[14]), "int list")	# Name y (line 3)
        self.assertEqual(str(id2type[16]), "int list")	# Name x (line 3)
        self.assertEqual(str(id2type[18]), "NoneType")	# AugAssign (line 4)
        self.assertEqual(str(id2type[19]), "int list")	# Name x (line 4)
        self.assertEqual(str(id2type[21]), "int list -> [int] -> int list")	# Add
        self.assertEqual(str(id2type[22]), "[int]")	# List (line 4)
        self.assertEqual(str(id2type[23]), "int")	# Num (line 4)
        self.assertEqual(str(id2type[25]), "int list")	# Return (line 5)
        self.assertEqual(str(id2type[26]), "int list")	# Name y (line 5)


    def test_mutable_attribute_assign(self):
        class Test():
            def __init__(self):
                self.a = [1, 2, 3]

            def forward(self):
                b = self.a
                b += [4]
                return self.a

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> int list")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "int list")	# Name b (line 2)
        self.assertEqual(str(id2type[8]), "int list")	# Attribute a (line 2)
        self.assertEqual(str(id2type[9]), "class Test")	# Name self (line 2)
        self.assertEqual(str(id2type[12]), "NoneType")	# AugAssign (line 3)
        self.assertEqual(str(id2type[13]), "int list")	# Name b (line 3)
        self.assertEqual(str(id2type[15]), "int list -> [int] -> int list")	# Add
        self.assertEqual(str(id2type[16]), "[int]")	# List (line 3)
        self.assertEqual(str(id2type[17]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[19]), "int list")	# Return (line 4)
        self.assertEqual(str(id2type[20]), "int list")	# Attribute a (line 4)
        self.assertEqual(str(id2type[21]), "class Test")	# Name self (line 4)


# ==============================================================================

def h(x, y):
    return x + y


class TestInline(unittest.TestCase):
    def test_calling_user_defined_function(self):
        # TODO(momohatt): h をここにもってきたい

        class Test():
            def forward(self, x):
                return h(x, 1)

        id2type = generate_id2type_from_forward(Test(), (1,))

        self.assertEqual(str(id2type[1]), "class Test -> int -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[8]), "int")	# Call (line 2)
        self.assertEqual(str(id2type[9]), "int -> int -> int")	# Name h (line 2)
        self.assertEqual(str(id2type[11]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[14]), "int -> int -> int")	# FunctionDef h (line 1)
        self.assertEqual(str(id2type[20]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[21]), "int")	# BinOp (line 2)
        self.assertEqual(str(id2type[22]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[24]), "int -> int -> int")	# Add
        self.assertEqual(str(id2type[25]), "int")	# Name y (line 2)


    def test_calling_user_defined_method(self):
        class A():
            def f(self, x):
                return x

        class Test():
            def __init__(self):
                self.a = A()

            def forward(self, x):
                return self.a.f(x)

        id2type = generate_id2type_from_forward(Test(), (1,))

        self.assertEqual(str(id2type[1]), "class Test -> int -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[8]), "int")	# Call (line 2)
        self.assertEqual(str(id2type[9]), "class A -> int -> int")	# Attribute f (line 2)
        self.assertEqual(str(id2type[10]), "class A")	# Attribute a (line 2)
        self.assertEqual(str(id2type[11]), "class Test")	# Name self (line 2)
        self.assertEqual(str(id2type[15]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[17]), "class A -> int -> int")	# FunctionDef f (line 1)
        self.assertEqual(str(id2type[23]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[24]), "int")	# Name x (line 2)


    def test_calling_user_defined_callable_class(self):
        class B():
            def __call__(self):
                return 1

        class Test():
            def __init__(self):
                self.b = B()

            def forward(self, x):
                return self.b() + x

        id2type = generate_id2type_from_forward(Test(), (1,))

        self.assertEqual(str(id2type[1]), "class Test -> int -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[8]), "int")	# BinOp (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Call (line 2)
        self.assertEqual(str(id2type[10]), "class B -> int")	# Attribute b (line 2)
        self.assertEqual(str(id2type[11]), "class Test")	# Name self (line 2)
        self.assertEqual(str(id2type[14]), "int -> int -> int")	# Add
        self.assertEqual(str(id2type[15]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[17]), "class B -> int")	# FunctionDef __call__ (line 1)
        self.assertEqual(str(id2type[21]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[22]), "int")	# Num (line 2)


    # TODO(hamaji): Run this test on CI.
    @pytest.mark.skip
    def test_calling_user_defined_callable_nested(self):
        class B():
            def f(self):
                return 1

            def __call__(self):
                return self.f()

        class Test():
            def __init__(self):
                self.b = B()

            def forward(self):
                return self.b()

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[6]), "int")	# Call (line 2)
        self.assertEqual(str(id2type[7]), "class B -> int")	# Attribute b (line 2)
        self.assertEqual(str(id2type[8]), "class Test")	# Name self (line 2)
        self.assertEqual(str(id2type[11]), "class B -> int")	# FunctionDef __call__ (line 1)
        self.assertEqual(str(id2type[15]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[16]), "int")	# Call (line 2)
        self.assertEqual(str(id2type[17]), "class B -> int")	# Attribute f (line 2)
        self.assertEqual(str(id2type[18]), "class B")	# Name self (line 2)
        self.assertEqual(str(id2type[21]), "class B -> int")	# FunctionDef f (line 1)
        self.assertEqual(str(id2type[25]), "int")	# Return (line 2)
        self.assertEqual(str(id2type[26]), "int")	# Num (line 2)



class TestLazy(unittest.TestCase):
    def test_lazy_init(self):
        class Test(chainer.Chain):
            def forward(self, x):
                if x is None:
                    x = 42
                return x

        id2type = generate_id2type_from_forward(Test(), (None,))

        self.assertEqual(str(id2type[1]), "class Test -> NoneType -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# If
        self.assertEqual(str(id2type[8]), "bool")	# Compare  (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign
        self.assertEqual(str(id2type[14]), "int")	# Name x (line 3)
        self.assertEqual(str(id2type[16]), "int")	# Num 42 (line 3)
        self.assertEqual(str(id2type[17]), "int")	# Return
        self.assertEqual(str(id2type[18]), "int")	# Name x (line 4)


    def test_lazy_init_branch_if(self):
        type_inference_tools.reset_state()

        class Test(chainer.Chain):
            def forward(self, x):
                if x is None:
                    x = 42
                else:
                    x += 1
                return x

        id2type = generate_id2type_from_forward(Test(), (None,))

        self.assertEqual(str(id2type[1]), "class Test -> NoneType -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# If
        self.assertEqual(str(id2type[8]), "bool")	# Compare  (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign
        self.assertEqual(str(id2type[14]), "int")	# Name x (line 3)
        self.assertEqual(str(id2type[16]), "int")	# Num 42 (line 3)
        self.assertEqual(str(id2type[17]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[18]), "a0 (from line 5)")	# Name x (line 5)
        self.assertEqual(str(id2type[20]), "NoneType -> int -> a0 (from line 5)")	# Add
        self.assertEqual(str(id2type[21]), "int")	# Num 1 (line 5)
        self.assertEqual(str(id2type[22]), "int")	# Return
        self.assertEqual(str(id2type[23]), "int")	# Name x (line 6)


    def test_lazy_init_branch_else(self):
        class Test(chainer.Chain):
            def forward(self, x):
                if x is None:
                    x = 42
                else:
                    x += 1
                return x

        # XXX: the input is different from the previous one
        id2type = generate_id2type_from_forward(Test(), (2,))

        self.assertEqual(str(id2type[1]), "class Test -> int -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# If
        self.assertEqual(str(id2type[8]), "bool")	# Compare  (line 2)
        self.assertEqual(str(id2type[9]), "int")	# Name x (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign
        self.assertEqual(str(id2type[14]), "int")	# Name x (line 3)
        self.assertEqual(str(id2type[16]), "int")	# Num 42 (line 3)
        self.assertEqual(str(id2type[17]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[18]), "int")	# Name x (line 5)
        self.assertEqual(str(id2type[20]), "int -> int -> int")	# Add
        self.assertEqual(str(id2type[21]), "int")	# Num 1 (line 5)
        self.assertEqual(str(id2type[22]), "int")	# Return
        self.assertEqual(str(id2type[23]), "int")	# Name x (line 6)


    def test_lazy_attribute_init(self):
        class Test():
            def __init__(self):
                self.y = None

            def forward(self):
                if self.y is None:
                    self.y = 42
                return self.y

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> int")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# If
        self.assertEqual(str(id2type[6]), "bool")	# Compare  (line 2)
        self.assertEqual(str(id2type[7]), "int")	# Attribute self.y (line 2)
        self.assertEqual(str(id2type[8]), "class Test")	# Name self (line 2)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign
        self.assertEqual(str(id2type[14]), "NoneType")	# Attribute self.y (line 3)
        self.assertEqual(str(id2type[15]), "class Test")	# Name self (line 3)
        self.assertEqual(str(id2type[18]), "int")	# Num 42 (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Return
        self.assertEqual(str(id2type[20]), "int")	# Attribute self.y (line 4)
        self.assertEqual(str(id2type[21]), "class Test")	# Name self (line 4)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
