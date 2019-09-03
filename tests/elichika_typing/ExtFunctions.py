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
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "ndarray(dtype(int64))")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "ndarray(dtype(int64))")	# Call (line 2)
        self.assertEqual(str(id2type[9]), "[int] -> ndarray(dtype(int64))")	# Attribute array (line 2)
        self.assertEqual(str(id2type[13]), "[int]")	# List (line 2)
        self.assertEqual(str(id2type[14]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[16]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[17]), "ndarray(dtype(float64))")	# Name y (line 3)
        self.assertEqual(str(id2type[19]), "ndarray(dtype(float64))")	# Call (line 3)
        self.assertEqual(str(id2type[20]), "int -> ndarray(dtype(float64))")	# Attribute array (line 3)
        self.assertEqual(str(id2type[24]), "int")	# Num (line 3)
        self.assertEqual(str(id2type[30]), "NoneType")	# Assign (line 4)
        self.assertEqual(str(id2type[31]), "ndarray(dtype(float32))")	# Name z (line 4)
        self.assertEqual(str(id2type[33]), "ndarray(dtype(float32))")	# Call (line 4)
        self.assertEqual(str(id2type[34]), "int -> ndarray(dtype(float32))")	# Attribute array (line 4)
        self.assertEqual(str(id2type[38]), "int")	# Num (line 4)
        self.assertEqual(str(id2type[41]), "NoneType")	# Assign (line 5)
        self.assertEqual(str(id2type[42]), "ndarray(dtype(float32))")	# Name w (line 5)
        self.assertEqual(str(id2type[44]), "ndarray(dtype(float32))")	# Call (line 5)
        self.assertEqual(str(id2type[45]), "string -> ndarray(dtype(float32))")	# Attribute astype (line 5)
        self.assertEqual(str(id2type[46]), "ndarray(dtype(float64))")	# Call (line 5)
        self.assertEqual(str(id2type[47]), "int -> ndarray(dtype(float64))")	# Attribute zeros (line 5)
        self.assertEqual(str(id2type[51]), "int")	# Num (line 5)
        self.assertEqual(str(id2type[53]), "string")	# Str (line 5)
        self.assertEqual(str(id2type[54]), "NoneType")	# Assign (line 6)
        self.assertEqual(str(id2type[55]), "ndarray(dtype(int32))")	# Name u (line 6)
        self.assertEqual(str(id2type[57]), "ndarray(dtype(int32))")	# Call (line 6)
        self.assertEqual(str(id2type[58]), "dtype(int32) -> ndarray(dtype(int32))")	# Attribute astype (line 6)
        self.assertEqual(str(id2type[59]), "ndarray(dtype(float64))")	# Call (line 6)
        self.assertEqual(str(id2type[60]), "int -> ndarray(dtype(float64))")	# Attribute zeros (line 6)
        self.assertEqual(str(id2type[64]), "int")	# Num (line 6)
        self.assertEqual(str(id2type[66]), "dtype(int32)")	# Attribute int32 (line 6)
        self.assertEqual(str(id2type[67]), "module")	# Name np (line 6)


    def test_numpy_zeros(self):
        class Test():
            def forward(self):
                x = np.zeros((3, 3))
                y = np.zeros(3, dtype='int64')

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "class Test -> NoneType")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "ndarray(dtype(float64))")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "ndarray(dtype(float64))")	# Call (line 2)
        self.assertEqual(str(id2type[9]), "(int, int) -> ndarray(dtype(float64))")	# Attribute zeros (line 2)
        self.assertEqual(str(id2type[13]), "(int, int)")	# Tuple (line 2)
        self.assertEqual(str(id2type[14]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[15]), "int")	# Num (line 2)
        self.assertEqual(str(id2type[17]), "NoneType")	# Assign (line 3)
        self.assertEqual(str(id2type[18]), "ndarray(dtype(int64))")	# Name y (line 3)
        self.assertEqual(str(id2type[20]), "ndarray(dtype(int64))")	# Call (line 3)
        self.assertEqual(str(id2type[21]), "int -> ndarray(dtype(int64))")	# Attribute zeros (line 3)
        self.assertEqual(str(id2type[25]), "int")	# Num (line 3)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
