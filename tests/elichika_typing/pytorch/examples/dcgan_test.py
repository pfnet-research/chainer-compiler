import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.examples.dcgan import gen_DCGAN_Generator_test
from testcases.pytorch.examples.dcgan import gen_DCGAN_Discriminator_test


class TestDCGAN(unittest.TestCase):
    def test_DCGAN_Generator(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_DCGAN_Generator_test()
        id2type = generate_id2type_from_forward(model, forward_args)

        self.assertEqual(str(id2type[1]), "class Generator -> torch.Tensor(float32, (5, 6, 1, 1)) -> torch.Tensor(float32, (5, 3, 64, 64))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (5, 3, 64, 64))")	# Call self.main(input) (line 6)
        self.assertEqual(str(id2type[12]), "class Generator")	# Name self (line 6)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (5, 6, 1, 1))")	# Name input (line 6)
        self.assertEqual(str(id2type[17]), "torch.Tensor(float32, (5, 3, 64, 64))")	# Return
        self.assertEqual(str(id2type[18]), "torch.Tensor(float32, (5, 3, 64, 64))")	# Name output (line 7)


    def test_DCGAN_Discriminator(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_DCGAN_Discriminator_test()
        id2type = generate_id2type_from_forward(model, forward_args)

        self.assertEqual(str(id2type[1]), "class Discriminator -> torch.Tensor(float32, (5, 3, 64, 64)) -> torch.Tensor(float32, (5,))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (5, 1, 1, 1))")	# Call self.main(input) (line 6)
        self.assertEqual(str(id2type[12]), "class Discriminator")	# Name self (line 6)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (5, 3, 64, 64))")	# Name input (line 6)
        self.assertEqual(str(id2type[17]), "torch.Tensor(float32, (5,))")	# Return
        self.assertEqual(str(id2type[18]), "torch.Tensor(float32, (5,))")	# Call output.view(-1, 1).squeeze(dim=1) (line 8)
        self.assertEqual(str(id2type[20]), "torch.Tensor(float32, (5, 1))")	# Call output.view(-1, 1) (line 8)
        self.assertEqual(str(id2type[22]), "torch.Tensor(float32, (5, 1, 1, 1))")	# Name output (line 8)
        self.assertEqual(str(id2type[25]), "int")	# UnaryOp -1 (line 8)
        self.assertEqual(str(id2type[27]), "int")	# Constant 1 (line 8)
        self.assertEqual(str(id2type[28]), "int")	# Constant 1 (line 8)
        self.assertEqual(str(id2type[31]), "int")	# Constant 1 (line 8)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
