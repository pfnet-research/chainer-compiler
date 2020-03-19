import ast, gast
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class TestNumpy(unittest.TestCase):
    def test_numpy_array(self):
        class Test():
            def forward(self):
                x = np.array([0])
                y = np.array(0, dtype=np.float64)
                z = np.array([0], dtype='float32')
                w = np.zeros(0).astype('f')
                u = np.zeros(0).astype(np.int32)

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign
        self.assertEqual(str(id2type[6]), "ndarray(int64, (1,))")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "ndarray(int64, (1,))")	# Call np.array([0]) (line 2)
        self.assertEqual(str(id2type[13]), "[int]")	# List [0] (line 2)
        self.assertEqual(str(id2type[14]), "int")	# Num 0 (line 2)
        self.assertEqual(str(id2type[16]), "NoneType")	# Assign
        self.assertEqual(str(id2type[17]), "ndarray(float64, ())")	# Name y (line 3)
        self.assertEqual(str(id2type[19]), "ndarray(float64, ())")	# Call np.array(0, dtype=np.float64) (line 3)
        self.assertEqual(str(id2type[24]), "int")	# Num 0 (line 3)
        self.assertEqual(str(id2type[26]), "dtype(float64)")	# Attribute np.float64 (line 3)
        self.assertEqual(str(id2type[30]), "NoneType")	# Assign
        self.assertEqual(str(id2type[31]), "ndarray(float32, (1,))")	# Name z (line 4)
        self.assertEqual(str(id2type[33]), "ndarray(float32, (1,))")	# Call np.array([0], dtype='float32') (line 4)
        self.assertEqual(str(id2type[38]), "[int]")	# List [0] (line 4)
        self.assertEqual(str(id2type[39]), "int")	# Num 0 (line 4)
        self.assertEqual(str(id2type[42]), "string")	# Str 'float32' (line 4)
        self.assertEqual(str(id2type[43]), "NoneType")	# Assign
        self.assertEqual(str(id2type[44]), "ndarray(float32, (0,))")	# Name w (line 5)
        self.assertEqual(str(id2type[46]), "ndarray(float32, (0,))")	# Call np.zeros(0).astype('f') (line 5)
        self.assertEqual(str(id2type[48]), "ndarray(float64, (0,))")	# Call np.zeros(0) (line 5)
        self.assertEqual(str(id2type[53]), "int")	# Num 0 (line 5)
        self.assertEqual(str(id2type[55]), "string")	# Str 'f' (line 5)
        self.assertEqual(str(id2type[56]), "NoneType")	# Assign
        self.assertEqual(str(id2type[57]), "ndarray(int32, (0,))")	# Name u (line 6)
        self.assertEqual(str(id2type[59]), "ndarray(int32, (0,))")	# Call np.zeros(0).astype(np.int32) (line 6)
        self.assertEqual(str(id2type[61]), "ndarray(float64, (0,))")	# Call np.zeros(0) (line 6)
        self.assertEqual(str(id2type[66]), "int")	# Num 0 (line 6)
        self.assertEqual(str(id2type[68]), "dtype(int32)")	# Attribute np.int32 (line 6)


    def test_numpy_zeros(self):
        class Test():
            def forward(self):
                x = np.zeros((3, 3))
                y = np.zeros(3, dtype='int64')

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign
        self.assertEqual(str(id2type[6]), "ndarray(float64, (3, 3))")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "ndarray(float64, (3, 3))")	# Call np.zeros((3, 3)) (line 2)
        self.assertEqual(str(id2type[13]), "(int, int)")	# Tuple (3, 3) (line 2)
        self.assertEqual(str(id2type[14]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[15]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[17]), "NoneType")	# Assign
        self.assertEqual(str(id2type[18]), "ndarray(int64, (3,))")	# Name y (line 3)
        self.assertEqual(str(id2type[20]), "ndarray(int64, (3,))")	# Call np.zeros(3dtype='int64') (line 3)
        self.assertEqual(str(id2type[25]), "int")	# Num 3 (line 3)
        self.assertEqual(str(id2type[27]), "string")	# Str 'int64' (line 3)


    def test_expand_dims(self):
        class Test():
            def forward(self):
                F.expand_dims(np.zeros((2, 3, 4)), 1)
                F.expand_dims(np.zeros((2, 3, 4)), -2)

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Expr
        self.assertEqual(str(id2type[6]), "Variable(float64, (2, 1, 3, 4))")	# Call F.expand_dims(np.zeros((2, 3, 4)), 1) (line 2)
        self.assertEqual(str(id2type[11]), "ndarray(float64, (2, 3, 4))")	# Call np.zeros((2, 3, 4)) (line 2)
        self.assertEqual(str(id2type[16]), "(int, int, int)")	# Tuple (2, 3, 4) (line 2)
        self.assertEqual(str(id2type[17]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[18]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Num 4 (line 2)
        self.assertEqual(str(id2type[21]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[22]), "NoneType")	# Expr
        self.assertEqual(str(id2type[23]), "Variable(float64, (2, 3, 1, 4))")	# Call F.expand_dims(np.zeros((2, 3, 4)), -2) (line 3)
        self.assertEqual(str(id2type[28]), "ndarray(float64, (2, 3, 4))")	# Call np.zeros((2, 3, 4)) (line 3)
        self.assertEqual(str(id2type[33]), "(int, int, int)")	# Tuple (2, 3, 4) (line 3)
        self.assertEqual(str(id2type[34]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[35]), "int")	# Num 3 (line 3)
        self.assertEqual(str(id2type[36]), "int")	# Num 4 (line 3)
        self.assertEqual(str(id2type[38]), "int")	# UnaryOp -2 (line 3)
        self.assertEqual(str(id2type[40]), "int")	# Num 2 (line 3)


    def test_squeeze(self):
        class Test():
            def forward(self):
                F.squeeze(np.zeros((2, 1, 1, 3)))
                F.squeeze(np.zeros((2, 1, 1, 3)), axis=2)
                F.squeeze(np.zeros((2, 1, 1, 3)), axis=(1,2))

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Expr
        self.assertEqual(str(id2type[6]), "Variable(float64, (2, 3))")	# Call F.squeeze(np.zeros((2, 1, 1, 3))) (line 2)
        self.assertEqual(str(id2type[11]), "ndarray(float64, (2, 1, 1, 3))")	# Call np.zeros((2, 1, 1, 3)) (line 2)
        self.assertEqual(str(id2type[16]), "(int, int, int, int)")	# Tuple (2, 1, 1, 3) (line 2)
        self.assertEqual(str(id2type[17]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[18]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[20]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[22]), "NoneType")	# Expr
        self.assertEqual(str(id2type[23]), "Variable(float64, (2, 1, 3))")	# Call F.squeeze(np.zeros((2, 1, 1, 3)), axis=2) (line 3)
        self.assertEqual(str(id2type[28]), "ndarray(float64, (2, 1, 1, 3))")	# Call np.zeros((2, 1, 1, 3)) (line 3)
        self.assertEqual(str(id2type[33]), "(int, int, int, int)")	# Tuple (2, 1, 1, 3) (line 3)
        self.assertEqual(str(id2type[34]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[35]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[36]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[37]), "int")	# Num 3 (line 3)
        self.assertEqual(str(id2type[40]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[41]), "NoneType")	# Expr
        self.assertEqual(str(id2type[42]), "Variable(float64, (2, 3))")	# Call F.squeeze(np.zeros((2, 1, 1, 3)), axis=(1, 2)) (line 4)
        self.assertEqual(str(id2type[47]), "ndarray(float64, (2, 1, 1, 3))")	# Call np.zeros((2, 1, 1, 3)) (line 4)
        self.assertEqual(str(id2type[52]), "(int, int, int, int)")	# Tuple (2, 1, 1, 3) (line 4)
        self.assertEqual(str(id2type[53]), "int")	# Num 2 (line 4)
        self.assertEqual(str(id2type[54]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[55]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[56]), "int")	# Num 3 (line 4)
        self.assertEqual(str(id2type[59]), "(int, int)")	# Tuple (1, 2) (line 4)
        self.assertEqual(str(id2type[60]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[61]), "int")	# Num 2 (line 4)


    def test_sum(self):
        class Test():
            def forward(self):
                F.sum(np.zeros((1, 2, 3)), axis=-1)
                F.sum(np.zeros((1, 2, 3)), axis=1, keepdims=True)
                F.sum(np.zeros((1, 2, 3)), keepdims=True)

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Expr
        self.assertEqual(str(id2type[6]), "Variable(float64, (1, 2))")	# Call F.sum(np.zeros((1, 2, 3)), axis=-1) (line 3)
        self.assertEqual(str(id2type[11]), "ndarray(float64, (1, 2, 3))")	# Call np.zeros((1, 2, 3)) (line 3)
        self.assertEqual(str(id2type[16]), "(int, int, int)")	# Tuple (1, 2, 3) (line 3)
        self.assertEqual(str(id2type[17]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[18]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Num 3 (line 3)
        self.assertEqual(str(id2type[22]), "int")	# UnaryOp -1 (line 3)
        self.assertEqual(str(id2type[24]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[25]), "NoneType")	# Expr
        self.assertEqual(str(id2type[26]), "Variable(float64, (1, 1, 3))")	# Call F.sum(np.zeros((1, 2, 3)), axis=1, keepdims=True) (line 4)
        self.assertEqual(str(id2type[31]), "ndarray(float64, (1, 2, 3))")	# Call np.zeros((1, 2, 3)) (line 4)
        self.assertEqual(str(id2type[36]), "(int, int, int)")	# Tuple (1, 2, 3) (line 4)
        self.assertEqual(str(id2type[37]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[38]), "int")	# Num 2 (line 4)
        self.assertEqual(str(id2type[39]), "int")	# Num 3 (line 4)
        self.assertEqual(str(id2type[42]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[44]), "bool")	# NameConstant True (line 4)
        self.assertEqual(str(id2type[45]), "NoneType")	# Expr
        self.assertEqual(str(id2type[46]), "Variable(float64, (1, 1, 1))")	# Call F.sum(np.zeros((1, 2, 3)), keepdims=True) (line 5)
        self.assertEqual(str(id2type[51]), "ndarray(float64, (1, 2, 3))")	# Call np.zeros((1, 2, 3)) (line 5)
        self.assertEqual(str(id2type[56]), "(int, int, int)")	# Tuple (1, 2, 3) (line 5)
        self.assertEqual(str(id2type[57]), "int")	# Num 1 (line 5)
        self.assertEqual(str(id2type[58]), "int")	# Num 2 (line 5)
        self.assertEqual(str(id2type[59]), "int")	# Num 3 (line 5)
        self.assertEqual(str(id2type[62]), "bool")	# NameConstant True (line 5)


    def test_separate(self):
        class Test():
            def forward(self):
                F.separate(np.zeros((3, 4, 5)), axis=0)

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Expr
        self.assertEqual(str(id2type[6]), "(Variable(float64, (4, 5)), Variable(float64, (4, 5)), Variable(float64, (4, 5)))")	# Call F.separate(np.zeros((3, 4, 5)), axis=0) (line 2)
        self.assertEqual(str(id2type[11]), "ndarray(float64, (3, 4, 5))")	# Call np.zeros((3, 4, 5)) (line 2)
        self.assertEqual(str(id2type[16]), "(int, int, int)")	# Tuple (3, 4, 5) (line 2)
        self.assertEqual(str(id2type[17]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[18]), "int")	# Num 4 (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Num 5 (line 2)
        self.assertEqual(str(id2type[22]), "int")	# Num 0 (line 2)


    def test_split_axis(self):
        class Test():
            def forward(self):
                F.split_axis(np.zeros((3, 4, 5)), 2, 1)

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Expr
        self.assertEqual(str(id2type[6]), "(Variable(float64, (3, 2, 5)), Variable(float64, (3, 2, 5)))")	# Call F.split_axis(np.zeros((3, 4, 5)), 2, 1) (line 3)
        self.assertEqual(str(id2type[11]), "ndarray(float64, (3, 4, 5))")	# Call np.zeros((3, 4, 5)) (line 3)
        self.assertEqual(str(id2type[16]), "(int, int, int)")	# Tuple (3, 4, 5) (line 3)
        self.assertEqual(str(id2type[17]), "int")	# Num 3 (line 3)
        self.assertEqual(str(id2type[18]), "int")	# Num 4 (line 3)
        self.assertEqual(str(id2type[19]), "int")	# Num 5 (line 3)
        self.assertEqual(str(id2type[21]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[22]), "int")	# Num 1 (line 3)


    def test_concat(self):
        pass


    def test_stack(self):
        pass


    def test_hstack(self):
        pass


    def test_vstack(self):
        class Test():
            def forward(self):
                F.vstack([np.zeros((1, 3, 4)), np.zeros((2, 3, 4))])

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Expr
        self.assertEqual(str(id2type[6]), "Variable(float64, (3, 3, 4))")	# Call F.vstack([np.zeros((1, 3, 4)), np.zeros((2, 3, 4))]) (line 2)
        self.assertEqual(str(id2type[11]), "[ndarray(float64, (1, 3, 4)), ndarray(float64, (2, 3, 4))]")	# List [np.zeros((1, 3, 4)), np.zeros((2, 3, 4))] (line 2)
        self.assertEqual(str(id2type[12]), "ndarray(float64, (1, 3, 4))")	# Call np.zeros((1, 3, 4)) (line 2)
        self.assertEqual(str(id2type[17]), "(int, int, int)")	# Tuple (1, 3, 4) (line 2)
        self.assertEqual(str(id2type[18]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[20]), "int")	# Num 4 (line 2)
        self.assertEqual(str(id2type[22]), "ndarray(float64, (2, 3, 4))")	# Call np.zeros((2, 3, 4)) (line 2)
        self.assertEqual(str(id2type[27]), "(int, int, int)")	# Tuple (2, 3, 4) (line 2)
        self.assertEqual(str(id2type[28]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[29]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[30]), "int")	# Num 4 (line 2)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
