import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.mnist             import gen_MNIST_model


class TestMNIST(unittest.TestCase):
    def test_MNIST(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_MNIST_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for MNIST ===
        self.assertEqual(str(id2type[1]), "class Net -> torch.Tensor(float32, (64, 1, 28, 28)) -> torch.Tensor(float32, (64, 10))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[8]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 2)
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Call self.conv1(x) (line 2)
        self.assertEqual(str(id2type[11]), "torch.Tensor(float32, (64, 1, 28, 28)) -> torch.Tensor(float32, (64, 32, 26, 26))")	# Attribute self.conv1 (line 2)
        self.assertEqual(str(id2type[12]), "class Net")	# Name self (line 2)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (64, 1, 28, 28))")	# Name x (line 2)
        self.assertEqual(str(id2type[17]), "NoneType")	# Assign
        self.assertEqual(str(id2type[18]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 3)
        self.assertEqual(str(id2type[20]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Call F.relu(x) (line 3)
        self.assertEqual(str(id2type[21]), "torch.Tensor(float32, (64, 32, 26, 26)) -> torch.Tensor(float32, (64, 32, 26, 26))")	# Attribute F.relu (line 3)
        self.assertEqual(str(id2type[25]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 3)
        self.assertEqual(str(id2type[27]), "NoneType")	# Assign
        self.assertEqual(str(id2type[28]), "torch.Tensor(float32, (64, 64, 24, 24))")	# Name x (line 4)
        self.assertEqual(str(id2type[30]), "torch.Tensor(float32, (64, 64, 24, 24))")	# Call self.conv2(x) (line 4)
        self.assertEqual(str(id2type[31]), "torch.Tensor(float32, (64, 32, 26, 26)) -> torch.Tensor(float32, (64, 64, 24, 24))")	# Attribute self.conv2 (line 4)
        self.assertEqual(str(id2type[32]), "class Net")	# Name self (line 4)
        self.assertEqual(str(id2type[35]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 4)
        self.assertEqual(str(id2type[37]), "NoneType")	# Assign
        self.assertEqual(str(id2type[38]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 5)
        self.assertEqual(str(id2type[40]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Call F.max_pool2d(x, 2) (line 5)
        self.assertEqual(str(id2type[41]), "torch.Tensor(float32, (64, 64, 24, 24)) -> int -> torch.Tensor(float32, (64, 64, 12, 12))")	# Attribute F.max_pool2d (line 5)
        self.assertEqual(str(id2type[45]), "torch.Tensor(float32, (64, 64, 24, 24))")	# Name x (line 5)
        self.assertEqual(str(id2type[47]), "int")	# Constant 2 (line 5)
        self.assertEqual(str(id2type[48]), "NoneType")	# Assign
        self.assertEqual(str(id2type[49]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 6)
        self.assertEqual(str(id2type[51]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Call self.dropout1(x) (line 6)
        self.assertEqual(str(id2type[52]), "torch.Tensor(float32, (64, 64, 12, 12)) -> torch.Tensor(float32, (64, 64, 12, 12))")	# Attribute self.dropout1 (line 6)
        self.assertEqual(str(id2type[53]), "class Net")	# Name self (line 6)
        self.assertEqual(str(id2type[56]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 6)
        self.assertEqual(str(id2type[58]), "NoneType")	# Assign
        self.assertEqual(str(id2type[59]), "torch.Tensor(float32, (64, 9216))")	# Name x (line 7)
        self.assertEqual(str(id2type[61]), "torch.Tensor(float32, (64, 9216))")	# Call torch.flatten(x, start_dim=1) (line 7)
        self.assertEqual(str(id2type[62]), "torch.Tensor(float32, (64, 64, 12, 12)) -> torch.Tensor(float32, (64, 9216))")	# Attribute torch.flatten (line 7)
        self.assertEqual(str(id2type[66]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 7)
        self.assertEqual(str(id2type[69]), "int")	# Constant 1 (line 7)
        self.assertEqual(str(id2type[70]), "NoneType")	# Assign
        self.assertEqual(str(id2type[71]), "torch.Tensor(float32, (64, 128))")	# Name x (line 8)
        self.assertEqual(str(id2type[73]), "torch.Tensor(float32, (64, 128))")	# Call self.fc1(x) (line 8)
        self.assertEqual(str(id2type[74]), "torch.Tensor(float32, (64, 9216)) -> torch.Tensor(float32, (64, 128))")	# Attribute self.fc1 (line 8)
        self.assertEqual(str(id2type[75]), "class Net")	# Name self (line 8)
        self.assertEqual(str(id2type[78]), "torch.Tensor(float32, (64, 9216))")	# Name x (line 8)
        self.assertEqual(str(id2type[80]), "NoneType")	# Assign
        self.assertEqual(str(id2type[81]), "torch.Tensor(float32, (64, 128))")	# Name x (line 9)
        self.assertEqual(str(id2type[83]), "torch.Tensor(float32, (64, 128))")	# Call F.relu(x) (line 9)
        self.assertEqual(str(id2type[84]), "torch.Tensor(float32, (64, 128)) -> torch.Tensor(float32, (64, 128))")	# Attribute F.relu (line 9)
        self.assertEqual(str(id2type[88]), "torch.Tensor(float32, (64, 128))")	# Name x (line 9)
        self.assertEqual(str(id2type[90]), "NoneType")	# Assign
        self.assertEqual(str(id2type[91]), "torch.Tensor(float32, (64, 128))")	# Name x (line 10)
        self.assertEqual(str(id2type[93]), "torch.Tensor(float32, (64, 128))")	# Call self.dropout2(x) (line 10)
        self.assertEqual(str(id2type[94]), "torch.Tensor(float32, (64, 128)) -> torch.Tensor(float32, (64, 128))")	# Attribute self.dropout2 (line 10)
        self.assertEqual(str(id2type[95]), "class Net")	# Name self (line 10)
        self.assertEqual(str(id2type[98]), "torch.Tensor(float32, (64, 128))")	# Name x (line 10)
        self.assertEqual(str(id2type[100]), "NoneType")	# Assign
        self.assertEqual(str(id2type[101]), "torch.Tensor(float32, (64, 10))")	# Name x (line 11)
        self.assertEqual(str(id2type[103]), "torch.Tensor(float32, (64, 10))")	# Call self.fc2(x) (line 11)
        self.assertEqual(str(id2type[104]), "torch.Tensor(float32, (64, 128)) -> torch.Tensor(float32, (64, 10))")	# Attribute self.fc2 (line 11)
        self.assertEqual(str(id2type[105]), "class Net")	# Name self (line 11)
        self.assertEqual(str(id2type[108]), "torch.Tensor(float32, (64, 128))")	# Name x (line 11)
        self.assertEqual(str(id2type[110]), "NoneType")	# Assign
        self.assertEqual(str(id2type[111]), "torch.Tensor(float32, (64, 10))")	# Name output (line 12)
        self.assertEqual(str(id2type[113]), "torch.Tensor(float32, (64, 10))")	# Call F.log_softmax(x, dim=1) (line 12)
        self.assertEqual(str(id2type[114]), "torch.Tensor(float32, (64, 10)) -> torch.Tensor(float32, (64, 10))")	# Attribute F.log_softmax (line 12)
        self.assertEqual(str(id2type[118]), "torch.Tensor(float32, (64, 10))")	# Name x (line 12)
        self.assertEqual(str(id2type[121]), "int")	# Constant 1 (line 12)
        self.assertEqual(str(id2type[122]), "torch.Tensor(float32, (64, 10))")	# Return
        self.assertEqual(str(id2type[123]), "torch.Tensor(float32, (64, 10))")	# Name output (line 13)
        # === END ASSERTIONS for MNIST ===

def main():
    unittest.main()

if __name__ == '__main__':
    main()
