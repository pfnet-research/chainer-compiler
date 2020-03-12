import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.examples.super_resolution import gen_SuperResolution_test


class TestSuperResolution(unittest.TestCase):
    def test_SuperResolution(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_SuperResolution_test()
        id2type = generate_id2type_from_forward(model, forward_args)

        self.assertEqual(str(id2type[1]), "class Net -> torch.Tensor(float32, (5, 1, 32, 32)) -> torch.Tensor(float32, (5, 1, 128, 128))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[8]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Name x (line 2)
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Call self.relu(self.conv1(x)) (line 2)
        self.assertEqual(str(id2type[11]), "torch.Tensor(float32, (5, 64, 32, 32)) -> torch.Tensor(float32, (5, 64, 32, 32))")	# Attribute self.relu (line 2)
        self.assertEqual(str(id2type[12]), "class Net")	# Name self (line 2)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Call self.conv1(x) (line 2)
        self.assertEqual(str(id2type[16]), "torch.Tensor(float32, (5, 1, 32, 32)) -> torch.Tensor(float32, (5, 64, 32, 32))")	# Attribute self.conv1 (line 2)
        self.assertEqual(str(id2type[17]), "class Net")	# Name self (line 2)
        self.assertEqual(str(id2type[20]), "torch.Tensor(float32, (5, 1, 32, 32))")	# Name x (line 2)
        self.assertEqual(str(id2type[22]), "NoneType")	# Assign
        self.assertEqual(str(id2type[23]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Name x (line 3)
        self.assertEqual(str(id2type[25]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Call self.relu(self.conv2(x)) (line 3)
        self.assertEqual(str(id2type[26]), "torch.Tensor(float32, (5, 64, 32, 32)) -> torch.Tensor(float32, (5, 64, 32, 32))")	# Attribute self.relu (line 3)
        self.assertEqual(str(id2type[27]), "class Net")	# Name self (line 3)
        self.assertEqual(str(id2type[30]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Call self.conv2(x) (line 3)
        self.assertEqual(str(id2type[31]), "torch.Tensor(float32, (5, 64, 32, 32)) -> torch.Tensor(float32, (5, 64, 32, 32))")	# Attribute self.conv2 (line 3)
        self.assertEqual(str(id2type[32]), "class Net")	# Name self (line 3)
        self.assertEqual(str(id2type[35]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Name x (line 3)
        self.assertEqual(str(id2type[37]), "NoneType")	# Assign
        self.assertEqual(str(id2type[38]), "torch.Tensor(float32, (5, 32, 32, 32))")	# Name x (line 4)
        self.assertEqual(str(id2type[40]), "torch.Tensor(float32, (5, 32, 32, 32))")	# Call self.relu(self.conv3(x)) (line 4)
        self.assertEqual(str(id2type[41]), "torch.Tensor(float32, (5, 32, 32, 32)) -> torch.Tensor(float32, (5, 32, 32, 32))")	# Attribute self.relu (line 4)
        self.assertEqual(str(id2type[42]), "class Net")	# Name self (line 4)
        self.assertEqual(str(id2type[45]), "torch.Tensor(float32, (5, 32, 32, 32))")	# Call self.conv3(x) (line 4)
        self.assertEqual(str(id2type[46]), "torch.Tensor(float32, (5, 64, 32, 32)) -> torch.Tensor(float32, (5, 32, 32, 32))")	# Attribute self.conv3 (line 4)
        self.assertEqual(str(id2type[47]), "class Net")	# Name self (line 4)
        self.assertEqual(str(id2type[50]), "torch.Tensor(float32, (5, 64, 32, 32))")	# Name x (line 4)
        self.assertEqual(str(id2type[52]), "NoneType")	# Assign
        self.assertEqual(str(id2type[53]), "torch.Tensor(float32, (5, 1, 128, 128))")	# Name x (line 5)
        self.assertEqual(str(id2type[55]), "torch.Tensor(float32, (5, 1, 128, 128))")	# Call self.pixel_shuffle(self.conv4(x)) (line 5)
        self.assertEqual(str(id2type[56]), "torch.Tensor(float32, (5, 16, 32, 32)) -> torch.Tensor(float32, (5, 1, 128, 128))")	# Attribute self.pixel_shuffle (line 5)
        self.assertEqual(str(id2type[57]), "class Net")	# Name self (line 5)
        self.assertEqual(str(id2type[60]), "torch.Tensor(float32, (5, 16, 32, 32))")	# Call self.conv4(x) (line 5)
        self.assertEqual(str(id2type[61]), "torch.Tensor(float32, (5, 32, 32, 32)) -> torch.Tensor(float32, (5, 16, 32, 32))")	# Attribute self.conv4 (line 5)
        self.assertEqual(str(id2type[62]), "class Net")	# Name self (line 5)
        self.assertEqual(str(id2type[65]), "torch.Tensor(float32, (5, 32, 32, 32))")	# Name x (line 5)
        self.assertEqual(str(id2type[67]), "torch.Tensor(float32, (5, 1, 128, 128))")	# Return
        self.assertEqual(str(id2type[68]), "torch.Tensor(float32, (5, 1, 128, 128))")	# Name x (line 6)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
