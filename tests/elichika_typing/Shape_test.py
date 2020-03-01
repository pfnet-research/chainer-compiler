import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from   chainer_compiler.elichika.testtools import generate_id2type_from_forward
from   chainer_compiler.elichika.typing import types

class TestShape(unittest.TestCase):
    def test_type_hints(self):
        class Test():
            def forward(self, x: types.TyNdarray(np.float32, ('a', 'b'))):
                h = F.split_axis(x, 2, 1)
                return h

        model, forward_args = Test(), (np.zeros((10, 10)).astype(np.float32),)
        id2type = generate_id2type_from_forward(model, forward_args)

        self.assertEqual(str(id2type[1]), "class Test -> ndarray(float32, (10 (a), 10 (b))) -> (Variable(float32, (10 (a), 5 (b // 2))), Variable(float32, (10 (a), 5 (b // 2))))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[20]), "NoneType")	# Assign
        self.assertEqual(str(id2type[21]), "(Variable(float32, (10 (a), 5 (b // 2))), Variable(float32, (10 (a), 5 (b // 2))))")	# Name h (line 2)
        self.assertEqual(str(id2type[23]), "(Variable(float32, (10 (a), 5 (b // 2))), Variable(float32, (10 (a), 5 (b // 2))))")	# Call F.split_axis(x, 2, 1) (line 2)
        self.assertEqual(str(id2type[24]), "ndarray(float32, (10 (a), 10 (b))) -> int -> int -> (Variable(float32, (10 (a), 5 (b // 2))), Variable(float32, (10 (a), 5 (b // 2))))")	# Attribute F.split_axis (line 2)
        self.assertEqual(str(id2type[28]), "ndarray(float32, (10 (a), 10 (b)))")	# Name x (line 2)
        self.assertEqual(str(id2type[30]), "int")	# Constant 2 (line 2)
        self.assertEqual(str(id2type[31]), "int")	# Constant 1 (line 2)
        self.assertEqual(str(id2type[32]), "(Variable(float32, (10 (a), 5 (b // 2))), Variable(float32, (10 (a), 5 (b // 2))))")	# Return
        self.assertEqual(str(id2type[33]), "(Variable(float32, (10 (a), 5 (b // 2))), Variable(float32, (10 (a), 5 (b // 2))))")	# Name h (line 3)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
