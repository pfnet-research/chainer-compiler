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
        # === function forward ===
        self.assertEqual(str(id2type[8]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 2)
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Call self.conv1(x) (line 2)
        self.assertEqual(str(id2type[12]), "class Net")	# Name self (line 2)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (64, 1, 28, 28))")	# Name x (line 2)
        self.assertEqual(str(id2type[18]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 3)
        self.assertEqual(str(id2type[20]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Call F.relu(x) (line 3)
        self.assertEqual(str(id2type[25]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 3)
        self.assertEqual(str(id2type[28]), "torch.Tensor(float32, (64, 64, 24, 24))")	# Name x (line 4)
        self.assertEqual(str(id2type[30]), "torch.Tensor(float32, (64, 64, 24, 24))")	# Call self.conv2(x) (line 4)
        self.assertEqual(str(id2type[32]), "class Net")	# Name self (line 4)
        self.assertEqual(str(id2type[35]), "torch.Tensor(float32, (64, 32, 26, 26))")	# Name x (line 4)
        self.assertEqual(str(id2type[38]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 5)
        self.assertEqual(str(id2type[40]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Call F.max_pool2d(x, 2) (line 5)
        self.assertEqual(str(id2type[45]), "torch.Tensor(float32, (64, 64, 24, 24))")	# Name x (line 5)
        self.assertEqual(str(id2type[47]), "int")	# Constant 2 (line 5)
        self.assertEqual(str(id2type[49]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 6)
        self.assertEqual(str(id2type[51]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Call self.dropout1(x) (line 6)
        self.assertEqual(str(id2type[53]), "class Net")	# Name self (line 6)
        self.assertEqual(str(id2type[56]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 6)
        self.assertEqual(str(id2type[59]), "torch.Tensor(float32, (64, 9216))")	# Name x (line 7)
        self.assertEqual(str(id2type[61]), "torch.Tensor(float32, (64, 9216))")	# Call torch.flatten(x, start_dim=1) (line 7)
        self.assertEqual(str(id2type[66]), "torch.Tensor(float32, (64, 64, 12, 12))")	# Name x (line 7)
        self.assertEqual(str(id2type[69]), "int")	# Constant 1 (line 7)
        self.assertEqual(str(id2type[71]), "torch.Tensor(float32, (64, 128))")	# Name x (line 8)
        self.assertEqual(str(id2type[73]), "torch.Tensor(float32, (64, 128))")	# Call self.fc1(x) (line 8)
        self.assertEqual(str(id2type[75]), "class Net")	# Name self (line 8)
        self.assertEqual(str(id2type[78]), "torch.Tensor(float32, (64, 9216))")	# Name x (line 8)
        self.assertEqual(str(id2type[81]), "torch.Tensor(float32, (64, 128))")	# Name x (line 9)
        self.assertEqual(str(id2type[83]), "torch.Tensor(float32, (64, 128))")	# Call F.relu(x) (line 9)
        self.assertEqual(str(id2type[88]), "torch.Tensor(float32, (64, 128))")	# Name x (line 9)
        self.assertEqual(str(id2type[91]), "torch.Tensor(float32, (64, 128))")	# Name x (line 10)
        self.assertEqual(str(id2type[93]), "torch.Tensor(float32, (64, 128))")	# Call self.dropout2(x) (line 10)
        self.assertEqual(str(id2type[95]), "class Net")	# Name self (line 10)
        self.assertEqual(str(id2type[98]), "torch.Tensor(float32, (64, 128))")	# Name x (line 10)
        self.assertEqual(str(id2type[101]), "torch.Tensor(float32, (64, 10))")	# Name x (line 11)
        self.assertEqual(str(id2type[103]), "torch.Tensor(float32, (64, 10))")	# Call self.fc2(x) (line 11)
        self.assertEqual(str(id2type[105]), "class Net")	# Name self (line 11)
        self.assertEqual(str(id2type[108]), "torch.Tensor(float32, (64, 128))")	# Name x (line 11)
        self.assertEqual(str(id2type[111]), "torch.Tensor(float32, (64, 10))")	# Name output (line 12)
        self.assertEqual(str(id2type[113]), "torch.Tensor(float32, (64, 10))")	# Call F.log_softmax(x, dim=1) (line 12)
        self.assertEqual(str(id2type[118]), "torch.Tensor(float32, (64, 10))")	# Name x (line 12)
        self.assertEqual(str(id2type[121]), "int")	# Constant 1 (line 12)
        self.assertEqual(str(id2type[123]), "torch.Tensor(float32, (64, 10))")	# Name output (line 13)
        # === END ASSERTIONS for MNIST ===

def main():
    unittest.main()

if __name__ == '__main__':
    main()
