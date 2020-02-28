import numpy as np
import pprint
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward

from testcases.elichika_tests.model.MLP import MLP
from testcases.elichika_tests.model.Resnet_with_loss import ResNet50
from testcases.ch2o_tests.model.GoogleNet_with_loss import GoogLeNet


def gen_MLP_model():
    out_n = 4
    batch_size = 100
    model = MLP(8, out_n)
    v = np.random.rand(batch_size, 3).astype(np.float32)
    w = np.random.randint(out_n, size=batch_size)
    forward_args = (v, w)

    return model, forward_args


def gen_ResNet50_model():
    model = ResNet50()
    bsize = 2
    v = np.random.rand(bsize, 3, 224, 224).astype(np.float32)
    t = np.random.randint(1000, size=bsize).astype(np.int32)
    forward_args = (v, t)

    return model, forward_args


def gen_GoogLeNet_model():
    model = GoogLeNet()
    v = np.random.rand(2, 3, 227, 227).astype(np.float32)
    t = np.random.randint(1000, size=2)

    return model, (v, t)


class TestMLP(unittest.TestCase):
    def test_MLP(self):
        model, forward_args = gen_MLP_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for MLP ===
        self.assertEqual(str(id2type[1]), "class MLP -> ndarray(float32, (100, 3)) -> ndarray(int64, (100,)) -> Variable(float32, ())")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[9]), "NoneType")	# Assign
        self.assertEqual(str(id2type[10]), "Variable(float32, (100, 8))")	# Name h1 (line 2)
        self.assertEqual(str(id2type[12]), "Variable(float32, (100, 8))")	# Call F.relu(self.l1(x)) (line 2)
        self.assertEqual(str(id2type[13]), "Variable(float32, (100, 8)) -> Variable(float32, (100, 8))")	# Attribute F.relu (line 2)
        self.assertEqual(str(id2type[17]), "Variable(float32, (100, 8))")	# Call self.l1(x) (line 2)
        self.assertEqual(str(id2type[18]), "ndarray(float32, (100, 3)) -> Variable(float32, (100, 8))")	# Attribute self.l1 (line 2)
        self.assertEqual(str(id2type[19]), "class MLP")	# Name self (line 2)
        self.assertEqual(str(id2type[22]), "ndarray(float32, (100, 3))")	# Name x (line 2)
        self.assertEqual(str(id2type[24]), "NoneType")	# Assign
        self.assertEqual(str(id2type[25]), "Variable(float32, (100, 8))")	# Name h2 (line 3)
        self.assertEqual(str(id2type[27]), "Variable(float32, (100, 8))")	# Call F.relu(self.l2(h1)) (line 3)
        self.assertEqual(str(id2type[28]), "Variable(float32, (100, 8)) -> Variable(float32, (100, 8))")	# Attribute F.relu (line 3)
        self.assertEqual(str(id2type[32]), "Variable(float32, (100, 8))")	# Call self.l2(h1) (line 3)
        self.assertEqual(str(id2type[33]), "Variable(float32, (100, 8)) -> Variable(float32, (100, 8))")	# Attribute self.l2 (line 3)
        self.assertEqual(str(id2type[34]), "class MLP")	# Name self (line 3)
        self.assertEqual(str(id2type[37]), "Variable(float32, (100, 8))")	# Name h1 (line 3)
        self.assertEqual(str(id2type[39]), "NoneType")	# Assign
        self.assertEqual(str(id2type[40]), "Variable(float32, (100, 4))")	# Name h3 (line 4)
        self.assertEqual(str(id2type[42]), "Variable(float32, (100, 4))")	# Call self.l3(h2) (line 4)
        self.assertEqual(str(id2type[43]), "Variable(float32, (100, 8)) -> Variable(float32, (100, 4))")	# Attribute self.l3 (line 4)
        self.assertEqual(str(id2type[44]), "class MLP")	# Name self (line 4)
        self.assertEqual(str(id2type[47]), "Variable(float32, (100, 8))")	# Name h2 (line 4)
        self.assertEqual(str(id2type[49]), "NoneType")	# Assign
        self.assertEqual(str(id2type[50]), "Variable(float32, ())")	# Name loss (line 5)
        self.assertEqual(str(id2type[52]), "Variable(float32, ())")	# Call F.softmax_cross_entropy(h3, t) (line 5)
        self.assertEqual(str(id2type[53]), "Variable(float32, (100, 4)) -> ndarray(int64, (100,)) -> Variable(float32, ())")	# Attribute F.softmax_cross_entropy (line 5)
        self.assertEqual(str(id2type[57]), "Variable(float32, (100, 4))")	# Name h3 (line 5)
        self.assertEqual(str(id2type[59]), "ndarray(int64, (100,))")	# Name t (line 5)
        self.assertEqual(str(id2type[61]), "Variable(float32, ())")	# Return
        self.assertEqual(str(id2type[62]), "Variable(float32, ())")	# Name loss (line 6)
        # === END ASSERTIONS for MLP ===


# class TestResNet50(unittest.TestCase):
#     def test_ResNet50(self):
#         model, forward_args = gen_ResNet50_model()
#         id2type = generate_id2type_from_forward(model, forward_args)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
