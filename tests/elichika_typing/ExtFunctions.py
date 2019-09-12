import ast, gast
import pprint
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
        self.assertEqual(str(id2type[6]), "ndarray(dtype=int64, shape=(1,))")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "ndarray(dtype=int64, shape=(1,))")	# Call np.array([0]) (line 2)
        self.assertEqual(str(id2type[9]), "[int] -> ndarray(dtype=int64, shape=(1,))")	# Attribute np.array (line 2)
        self.assertEqual(str(id2type[13]), "[int]")	# List [0] (line 2)
        self.assertEqual(str(id2type[14]), "int")	# Num 0 (line 2)
        self.assertEqual(str(id2type[16]), "NoneType")	# Assign
        self.assertEqual(str(id2type[17]), "ndarray(dtype=float64, shape=())")	# Name y (line 3)
        self.assertEqual(str(id2type[19]), "ndarray(dtype=float64, shape=())")	# Call np.array(0dtype=np.float64) (line 3)
        self.assertEqual(str(id2type[20]), "int -> ndarray(dtype=float64, shape=())")	# Attribute np.array (line 3)
        self.assertEqual(str(id2type[24]), "int")	# Num 0 (line 3)
        self.assertEqual(str(id2type[26]), "dtype(float64)")	# Attribute np.float64 (line 3)
        self.assertEqual(str(id2type[30]), "NoneType")	# Assign
        self.assertEqual(str(id2type[31]), "ndarray(dtype=float32, shape=(1,))")	# Name z (line 4)
        self.assertEqual(str(id2type[33]), "ndarray(dtype=float32, shape=(1,))")	# Call np.array([0]dtype='float32') (line 4)
        self.assertEqual(str(id2type[34]), "[int] -> ndarray(dtype=float32, shape=(1,))")	# Attribute np.array (line 4)
        self.assertEqual(str(id2type[38]), "[int]")	# List [0] (line 4)
        self.assertEqual(str(id2type[39]), "int")	# Num 0 (line 4)
        self.assertEqual(str(id2type[42]), "string")	# Str 'float32' (line 4)
        self.assertEqual(str(id2type[43]), "NoneType")	# Assign
        self.assertEqual(str(id2type[44]), "ndarray(dtype=float32, shape=0)")	# Name w (line 5)
        self.assertEqual(str(id2type[46]), "ndarray(dtype=float32, shape=0)")	# Call np.zeros(0).astype('f') (line 5)
        self.assertEqual(str(id2type[47]), "string -> ndarray(dtype=float32, shape=0)")	# Attribute np.zeros(0).astype (line 5)
        self.assertEqual(str(id2type[48]), "ndarray(dtype=float64, shape=0)")	# Call np.zeros(0) (line 5)
        self.assertEqual(str(id2type[49]), "int -> ndarray(dtype=float64, shape=0)")	# Attribute np.zeros (line 5)
        self.assertEqual(str(id2type[53]), "int")	# Num 0 (line 5)
        self.assertEqual(str(id2type[55]), "string")	# Str 'f' (line 5)
        self.assertEqual(str(id2type[56]), "NoneType")	# Assign
        self.assertEqual(str(id2type[57]), "ndarray(dtype=int32, shape=0)")	# Name u (line 6)
        self.assertEqual(str(id2type[59]), "ndarray(dtype=int32, shape=0)")	# Call np.zeros(0).astype(np.int32) (line 6)
        self.assertEqual(str(id2type[60]), "dtype(int32) -> ndarray(dtype=int32, shape=0)")	# Attribute np.zeros(0).astype (line 6)
        self.assertEqual(str(id2type[61]), "ndarray(dtype=float64, shape=0)")	# Call np.zeros(0) (line 6)
        self.assertEqual(str(id2type[62]), "int -> ndarray(dtype=float64, shape=0)")	# Attribute np.zeros (line 6)
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
        self.assertEqual(str(id2type[6]), "ndarray(dtype=float64, shape=(3, 3))")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "ndarray(dtype=float64, shape=(3, 3))")	# Call np.zeros((3, 3)) (line 2)
        self.assertEqual(str(id2type[9]), "(int, int) -> ndarray(dtype=float64, shape=(3, 3))")	# Attribute np.zeros (line 2)
        self.assertEqual(str(id2type[13]), "(int, int)")	# Tuple (3, 3) (line 2)
        self.assertEqual(str(id2type[14]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[15]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[17]), "NoneType")	# Assign
        self.assertEqual(str(id2type[18]), "ndarray(dtype=int64, shape=3)")	# Name y (line 3)
        self.assertEqual(str(id2type[20]), "ndarray(dtype=int64, shape=3)")	# Call np.zeros(3dtype='int64') (line 3)
        self.assertEqual(str(id2type[21]), "int -> ndarray(dtype=int64, shape=3)")	# Attribute np.zeros (line 3)
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
        self.assertEqual(str(id2type[6]), "Variable(dtype=float64, shape=(2, 1, 3, 4))")	# Call F.expand_dims(np.zeros((2, 3, 4)), 1) (line 2)
        self.assertEqual(str(id2type[7]), "ndarray(dtype=float64, shape=(2, 3, 4)) -> int -> Variable(dtype=float64, shape=(2, 1, 3, 4))")	# Attribute F.expand_dims (line 2)
        self.assertEqual(str(id2type[11]), "ndarray(dtype=float64, shape=(2, 3, 4))")	# Call np.zeros((2, 3, 4)) (line 2)
        self.assertEqual(str(id2type[12]), "(int, int, int) -> ndarray(dtype=float64, shape=(2, 3, 4))")	# Attribute np.zeros (line 2)
        self.assertEqual(str(id2type[16]), "(int, int, int)")	# Tuple (2, 3, 4) (line 2)
        self.assertEqual(str(id2type[17]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[18]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Num 4 (line 2)
        self.assertEqual(str(id2type[21]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[22]), "NoneType")	# Expr
        self.assertEqual(str(id2type[23]), "Variable(dtype=float64, shape=(2, 3, 1, 4))")	# Call F.expand_dims(np.zeros((2, 3, 4)), -2) (line 3)
        self.assertEqual(str(id2type[24]), "ndarray(dtype=float64, shape=(2, 3, 4)) -> int -> Variable(dtype=float64, shape=(2, 3, 1, 4))")	# Attribute F.expand_dims (line 3)
        self.assertEqual(str(id2type[28]), "ndarray(dtype=float64, shape=(2, 3, 4))")	# Call np.zeros((2, 3, 4)) (line 3)
        self.assertEqual(str(id2type[29]), "(int, int, int) -> ndarray(dtype=float64, shape=(2, 3, 4))")	# Attribute np.zeros (line 3)
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
        self.assertEqual(str(id2type[6]), "Variable(dtype=float64, shape=(2, 3))")	# Call F.squeeze(np.zeros((2, 1, 1, 3))) (line 2)
        self.assertEqual(str(id2type[7]), "ndarray(dtype=float64, shape=(2, 1, 1, 3)) -> Variable(dtype=float64, shape=(2, 3))")	# Attribute F.squeeze (line 2)
        self.assertEqual(str(id2type[11]), "ndarray(dtype=float64, shape=(2, 1, 1, 3))")	# Call np.zeros((2, 1, 1, 3)) (line 2)
        self.assertEqual(str(id2type[12]), "(int, int, int, int) -> ndarray(dtype=float64, shape=(2, 1, 1, 3))")	# Attribute np.zeros (line 2)
        self.assertEqual(str(id2type[16]), "(int, int, int, int)")	# Tuple (2, 1, 1, 3) (line 2)
        self.assertEqual(str(id2type[17]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[18]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[20]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[22]), "NoneType")	# Expr
        self.assertEqual(str(id2type[23]), "Variable(dtype=float64, shape=(2, 1, 3))")	# Call F.squeeze(np.zeros((2, 1, 1, 3)), axis=2) (line 3)
        self.assertEqual(str(id2type[24]), "ndarray(dtype=float64, shape=(2, 1, 1, 3)) -> Variable(dtype=float64, shape=(2, 1, 3))")	# Attribute F.squeeze (line 3)
        self.assertEqual(str(id2type[28]), "ndarray(dtype=float64, shape=(2, 1, 1, 3))")	# Call np.zeros((2, 1, 1, 3)) (line 3)
        self.assertEqual(str(id2type[29]), "(int, int, int, int) -> ndarray(dtype=float64, shape=(2, 1, 1, 3))")	# Attribute np.zeros (line 3)
        self.assertEqual(str(id2type[33]), "(int, int, int, int)")	# Tuple (2, 1, 1, 3) (line 3)
        self.assertEqual(str(id2type[34]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[35]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[36]), "int")	# Num 1 (line 3)
        self.assertEqual(str(id2type[37]), "int")	# Num 3 (line 3)
        self.assertEqual(str(id2type[40]), "int")	# Num 2 (line 3)
        self.assertEqual(str(id2type[41]), "NoneType")	# Expr
        self.assertEqual(str(id2type[42]), "Variable(dtype=float64, shape=(2, 3))")	# Call F.squeeze(np.zeros((2, 1, 1, 3)), axis=(1, 2)) (line 4)
        self.assertEqual(str(id2type[43]), "ndarray(dtype=float64, shape=(2, 1, 1, 3)) -> Variable(dtype=float64, shape=(2, 3))")	# Attribute F.squeeze (line 4)
        self.assertEqual(str(id2type[47]), "ndarray(dtype=float64, shape=(2, 1, 1, 3))")	# Call np.zeros((2, 1, 1, 3)) (line 4)
        self.assertEqual(str(id2type[48]), "(int, int, int, int) -> ndarray(dtype=float64, shape=(2, 1, 1, 3))")	# Attribute np.zeros (line 4)
        self.assertEqual(str(id2type[52]), "(int, int, int, int)")	# Tuple (2, 1, 1, 3) (line 4)
        self.assertEqual(str(id2type[53]), "int")	# Num 2 (line 4)
        self.assertEqual(str(id2type[54]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[55]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[56]), "int")	# Num 3 (line 4)
        self.assertEqual(str(id2type[59]), "(int, int)")	# Tuple (1, 2) (line 4)
        self.assertEqual(str(id2type[60]), "int")	# Num 1 (line 4)
        self.assertEqual(str(id2type[61]), "int")	# Num 2 (line 4)


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
        self.assertEqual(str(id2type[6]), "Variable(dtype=float64, shape=(3, 3, 4))")	# Call F.vstack([np.zeros((1, 3, 4)), np.zeros((2, 3, 4))]) (line 2)
        self.assertEqual(str(id2type[7]), "[ndarray(dtype=float64, shape=(1, 3, 4)), ndarray(dtype=float64, shape=(2, 3, 4))] -> Variable(dtype=float64, shape=(3, 3, 4))")	# Attribute F.vstack (line 2)
        self.assertEqual(str(id2type[11]), "[ndarray(dtype=float64, shape=(1, 3, 4)), ndarray(dtype=float64, shape=(2, 3, 4))]")	# List [np.zeros((1, 3, 4)), np.zeros((2, 3, 4))] (line 2)
        self.assertEqual(str(id2type[12]), "ndarray(dtype=float64, shape=(1, 3, 4))")	# Call np.zeros((1, 3, 4)) (line 2)
        self.assertEqual(str(id2type[13]), "(int, int, int) -> ndarray(dtype=float64, shape=(1, 3, 4))")	# Attribute np.zeros (line 2)
        self.assertEqual(str(id2type[17]), "(int, int, int)")	# Tuple (1, 3, 4) (line 2)
        self.assertEqual(str(id2type[18]), "int")	# Num 1 (line 2)
        self.assertEqual(str(id2type[19]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[20]), "int")	# Num 4 (line 2)
        self.assertEqual(str(id2type[22]), "ndarray(dtype=float64, shape=(2, 3, 4))")	# Call np.zeros((2, 3, 4)) (line 2)
        self.assertEqual(str(id2type[23]), "(int, int, int) -> ndarray(dtype=float64, shape=(2, 3, 4))")	# Attribute np.zeros (line 2)
        self.assertEqual(str(id2type[27]), "(int, int, int)")	# Tuple (2, 3, 4) (line 2)
        self.assertEqual(str(id2type[28]), "int")	# Num 2 (line 2)
        self.assertEqual(str(id2type[29]), "int")	# Num 3 (line 2)
        self.assertEqual(str(id2type[30]), "int")	# Num 4 (line 2)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
