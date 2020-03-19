import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.examples.mnist_hogwild import gen_MNIST_hogwild_test


class TestMNIST_hogwild(unittest.TestCase):
    def test_MNIST_hogwild(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_MNIST_hogwild_test()
        id2type = generate_id2type_from_forward(model, forward_args)

        self.assertEqual(str(id2type[1]), "class Net -> torch.Tensor(float32, (64, 1, 28, 28)) -> torch.Tensor(float32, (64, 10))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (64, 10, 12, 12))")	# Call F.relu(F.max_pool2d(self.conv1(x), 2)) (line 2)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (64, 10, 12, 12))")	# Call F.max_pool2d(self.conv1(x), 2) (line 2)
        self.assertEqual(str(id2type[20]), "torch.Tensor(float32, (64, 10, 24, 24))")	# Call self.conv1(x) (line 2)
        self.assertEqual(str(id2type[22]), "class Net")	# Name self (line 2)
        self.assertEqual(str(id2type[25]), "torch.Tensor(float32, (64, 1, 28, 28))")	# Name x (line 2)
        self.assertEqual(str(id2type[27]), "int")	# Constant 2 (line 2)
        self.assertEqual(str(id2type[28]), "NoneType")	# Assign
        self.assertEqual(str(id2type[31]), "torch.Tensor(float32, (64, 20, 4, 4))")	# Call F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) (line 3)
        self.assertEqual(str(id2type[36]), "torch.Tensor(float32, (64, 20, 4, 4))")	# Call F.max_pool2d(self.conv2_drop(self.conv2(x)), 2) (line 3)
        self.assertEqual(str(id2type[41]), "torch.Tensor(float32, (64, 20, 8, 8))")	# Call self.conv2_drop(self.conv2(x)) (line 3)
        self.assertEqual(str(id2type[43]), "class Net")	# Name self (line 3)
        self.assertEqual(str(id2type[46]), "torch.Tensor(float32, (64, 20, 8, 8))")	# Call self.conv2(x) (line 3)
        self.assertEqual(str(id2type[48]), "class Net")	# Name self (line 3)
        self.assertEqual(str(id2type[51]), "torch.Tensor(float32, (64, 10, 12, 12))")	# Name x (line 3)
        self.assertEqual(str(id2type[53]), "int")	# Constant 2 (line 3)
        self.assertEqual(str(id2type[54]), "NoneType")	# Assign
        self.assertEqual(str(id2type[57]), "torch.Tensor(float32, (64, 320))")	# Call x.view(-1, 320) (line 4)
        self.assertEqual(str(id2type[59]), "torch.Tensor(float32, (64, 20, 4, 4))")	# Name x (line 4)
        self.assertEqual(str(id2type[62]), "int")	# UnaryOp -1 (line 4)
        self.assertEqual(str(id2type[64]), "int")	# Constant 1 (line 4)
        self.assertEqual(str(id2type[65]), "int")	# Constant 320 (line 4)
        self.assertEqual(str(id2type[66]), "NoneType")	# Assign
        self.assertEqual(str(id2type[69]), "torch.Tensor(float32, (64, 50))")	# Call F.relu(self.fc1(x)) (line 5)
        self.assertEqual(str(id2type[74]), "torch.Tensor(float32, (64, 50))")	# Call self.fc1(x) (line 5)
        self.assertEqual(str(id2type[76]), "class Net")	# Name self (line 5)
        self.assertEqual(str(id2type[79]), "torch.Tensor(float32, (64, 320))")	# Name x (line 5)
        self.assertEqual(str(id2type[81]), "NoneType")	# Assign
        self.assertEqual(str(id2type[84]), "torch.Tensor(float32, (64, 50))")	# Call F.dropout(x, training=self.training) (line 6)
        self.assertEqual(str(id2type[89]), "torch.Tensor(float32, (64, 50))")	# Name x (line 6)
        self.assertEqual(str(id2type[92]), "bool")	# Attribute self.training (line 6)
        self.assertEqual(str(id2type[93]), "class Net")	# Name self (line 6)
        self.assertEqual(str(id2type[96]), "NoneType")	# Assign
        self.assertEqual(str(id2type[99]), "torch.Tensor(float32, (64, 10))")	# Call self.fc2(x) (line 7)
        self.assertEqual(str(id2type[101]), "class Net")	# Name self (line 7)
        self.assertEqual(str(id2type[104]), "torch.Tensor(float32, (64, 50))")	# Name x (line 7)
        self.assertEqual(str(id2type[106]), "torch.Tensor(float32, (64, 10))")	# Return
        self.assertEqual(str(id2type[107]), "torch.Tensor(float32, (64, 10))")	# Call F.log_softmax(x, dim=1) (line 8)
        self.assertEqual(str(id2type[112]), "torch.Tensor(float32, (64, 10))")	# Name x (line 8)
        self.assertEqual(str(id2type[115]), "int")	# Constant 1 (line 8)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
