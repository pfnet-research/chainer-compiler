import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.alexnet           import gen_AlexNet_model


class TestAlexNet(unittest.TestCase):
    def test_AlexNet(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_AlexNet_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for AlexNet ===
        self.assertEqual(str(id2type[1]), "class AlexNet -> torch.Tensor(float32, (1, 3, 224, 224)) -> torch.Tensor(float32, (1, 1000))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[8]), "torch.Tensor(float32, (1, 256, 6, 6))")	# Name x (line 2)
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (1, 256, 6, 6))")	# Call self.features(x) (line 2)
        self.assertEqual(str(id2type[11]), "torch.Tensor(float32, (1, 3, 224, 224)) -> torch.Tensor(float32, (1, 256, 6, 6))")	# Attribute self.features (line 2)
        self.assertEqual(str(id2type[12]), "class AlexNet")	# Name self (line 2)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (1, 3, 224, 224))")	# Name x (line 2)
        self.assertEqual(str(id2type[17]), "NoneType")	# Assign
        self.assertEqual(str(id2type[18]), "torch.Tensor(float32, (1, 256, 6, 6))")	# Name x (line 3)
        self.assertEqual(str(id2type[20]), "torch.Tensor(float32, (1, 256, 6, 6))")	# Call self.avgpool(x) (line 3)
        self.assertEqual(str(id2type[21]), "torch.Tensor(float32, (1, 256, 6, 6)) -> torch.Tensor(float32, (1, 256, 6, 6))")	# Attribute self.avgpool (line 3)
        self.assertEqual(str(id2type[22]), "class AlexNet")	# Name self (line 3)
        self.assertEqual(str(id2type[25]), "torch.Tensor(float32, (1, 256, 6, 6))")	# Name x (line 3)
        self.assertEqual(str(id2type[27]), "NoneType")	# Assign
        self.assertEqual(str(id2type[28]), "torch.Tensor(float32, (1, 9216))")	# Name x (line 4)
        self.assertEqual(str(id2type[30]), "torch.Tensor(float32, (1, 9216))")	# Call torch.flatten(x, start_dim=1) (line 4)
        self.assertEqual(str(id2type[31]), "torch.Tensor(float32, (1, 256, 6, 6)) -> torch.Tensor(float32, (1, 9216))")	# Attribute torch.flatten (line 4)
        self.assertEqual(str(id2type[35]), "torch.Tensor(float32, (1, 256, 6, 6))")	# Name x (line 4)
        self.assertEqual(str(id2type[38]), "int")	# Constant 1 (line 4)
        self.assertEqual(str(id2type[39]), "NoneType")	# Assign
        self.assertEqual(str(id2type[40]), "torch.Tensor(float32, (1, 1000))")	# Name x (line 5)
        self.assertEqual(str(id2type[42]), "torch.Tensor(float32, (1, 1000))")	# Call self.classifier(x) (line 5)
        self.assertEqual(str(id2type[43]), "torch.Tensor(float32, (1, 9216)) -> torch.Tensor(float32, (1, 1000))")	# Attribute self.classifier (line 5)
        self.assertEqual(str(id2type[44]), "class AlexNet")	# Name self (line 5)
        self.assertEqual(str(id2type[47]), "torch.Tensor(float32, (1, 9216))")	# Name x (line 5)
        self.assertEqual(str(id2type[49]), "torch.Tensor(float32, (1, 1000))")	# Return
        self.assertEqual(str(id2type[50]), "torch.Tensor(float32, (1, 1000))")	# Name x (line 6)
        # === END ASSERTIONS for AlexNet ===


def main():
    unittest.main()

if __name__ == '__main__':
    main()
