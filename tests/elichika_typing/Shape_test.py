import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import torch

from   chainer_compiler.elichika.testtools import generate_id2type_from_forward
from   chainer_compiler.elichika.typing import types

class TestShape(unittest.TestCase):
    def test_shape_unify(self):
        class Test():
            def forward(self, x):
                for i in range(5):
                    x = x.expand(4, x.size(1) + 1)
                return x

        model, forward_args = Test(), (torch.ones(4, 1, dtype=torch.float),)
        id2type = generate_id2type_from_forward(model, forward_args)

        self.assertEqual(str(id2type[1]), "class Test -> torch.Tensor(float32, (4, 1)) -> torch.Tensor(float32, (4, None))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# For
        self.assertEqual(str(id2type[8]), "int")	# Name i (line 2)
        self.assertEqual(str(id2type[10]), "int list")	# Call range(5) (line 2)
        self.assertEqual(str(id2type[11]), "int -> int list")	# Name range (line 2)
        self.assertEqual(str(id2type[13]), "int")	# Constant 5 (line 2)
        self.assertEqual(str(id2type[14]), "NoneType")	# Assign
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (4, None))")	# Name x (line 3)
        self.assertEqual(str(id2type[17]), "torch.Tensor(float32, (4, None))")	# Call x.expand(4, x.size(1) + 1) (line 3)
        self.assertEqual(str(id2type[18]), "int -> int -> torch.Tensor(float32, (4, None))")	# Attribute x.expand (line 3)
        self.assertEqual(str(id2type[19]), "torch.Tensor(float32, (4, None))")	# Name x (line 3)
        self.assertEqual(str(id2type[22]), "int")	# Constant 4 (line 3)
        self.assertEqual(str(id2type[23]), "int")	# BinOp x.size(1) + 1 (line 3)
        self.assertEqual(str(id2type[24]), "int")	# Call x.size(1) (line 3)
        self.assertEqual(str(id2type[25]), "int -> int")	# Attribute x.size (line 3)
        self.assertEqual(str(id2type[26]), "torch.Tensor(float32, (4, None))")	# Name x (line 3)
        self.assertEqual(str(id2type[29]), "int")	# Constant 1 (line 3)
        self.assertEqual(str(id2type[30]), "int -> int -> int")	# Add
        self.assertEqual(str(id2type[31]), "int")	# Constant 1 (line 3)
        self.assertEqual(str(id2type[32]), "torch.Tensor(float32, (4, None))")	# Return
        self.assertEqual(str(id2type[33]), "torch.Tensor(float32, (4, None))")	# Name x (line 4)


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
