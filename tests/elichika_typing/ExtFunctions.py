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
                x = np.array([4.0])
                return x

        id2type = generate_id2type_from_forward(Test(), ())

        self.assertEqual(str(id2type[1]), "Test -> ndarray(dtype=float64)")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[5]), "NoneType")	# Assign (line 2)
        self.assertEqual(str(id2type[6]), "ndarray(dtype=float64)")	# Name x (line 2)
        self.assertEqual(str(id2type[8]), "ndarray(dtype=float64)")	# Call (line 2)
        self.assertEqual(str(id2type[9]), "[float] -> ndarray(dtype=float64)")	# Attribute array (line 2)
        self.assertEqual(str(id2type[13]), "[float]")	# List (line 2)
        self.assertEqual(str(id2type[14]), "float")	# Num (line 2)
        self.assertEqual(str(id2type[16]), "ndarray(dtype=float64)")	# Return (line 3)
        self.assertEqual(str(id2type[17]), "ndarray(dtype=float64)")	# Name x (line 3)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
