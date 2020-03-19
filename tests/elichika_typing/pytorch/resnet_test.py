import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.resnet            import gen_ResNet18_model


class TestResNet(unittest.TestCase):
    def test_ResNet18(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_ResNet18_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for ResNet18 ===
        self.assertEqual(str(id2type[1]), "class ResNet -> torch.Tensor(float32, (1, 3, 224, 224)) -> torch.Tensor(float32, (1, 1000))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "torch.Tensor(float32, (1, 1000))")	# Return
        self.assertEqual(str(id2type[8]), "torch.Tensor(float32, (1, 1000))")	# Call self._forward_impl(x) (line 2)
        self.assertEqual(str(id2type[10]), "class ResNet")	# Name self (line 2)
        self.assertEqual(str(id2type[13]), "torch.Tensor(float32, (1, 3, 224, 224))")	# Name x (line 2)
        self.assertEqual(str(id2type[15]), "class ResNet -> torch.Tensor(float32, (1, 3, 224, 224)) -> torch.Tensor(float32, (1, 1000))")	# FunctionDef _forward_impl (line 1)
        self.assertEqual(str(id2type[21]), "NoneType")	# Assign
        self.assertEqual(str(id2type[22]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 3)
        self.assertEqual(str(id2type[24]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Call self.conv1(x) (line 3)
        self.assertEqual(str(id2type[26]), "class ResNet")	# Name self (line 3)
        self.assertEqual(str(id2type[29]), "torch.Tensor(float32, (1, 3, 224, 224))")	# Name x (line 3)
        self.assertEqual(str(id2type[31]), "NoneType")	# Assign
        self.assertEqual(str(id2type[32]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 4)
        self.assertEqual(str(id2type[34]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Call self.bn1(x) (line 4)
        self.assertEqual(str(id2type[36]), "class ResNet")	# Name self (line 4)
        self.assertEqual(str(id2type[39]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 4)
        self.assertEqual(str(id2type[41]), "NoneType")	# Assign
        self.assertEqual(str(id2type[42]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 5)
        self.assertEqual(str(id2type[44]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Call self.relu(x) (line 5)
        self.assertEqual(str(id2type[46]), "class ResNet")	# Name self (line 5)
        self.assertEqual(str(id2type[49]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 5)
        self.assertEqual(str(id2type[51]), "NoneType")	# Assign
        self.assertEqual(str(id2type[52]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 6)
        self.assertEqual(str(id2type[54]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.maxpool(x) (line 6)
        self.assertEqual(str(id2type[56]), "class ResNet")	# Name self (line 6)
        self.assertEqual(str(id2type[59]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 6)
        self.assertEqual(str(id2type[61]), "NoneType")	# Assign
        self.assertEqual(str(id2type[62]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 8)
        self.assertEqual(str(id2type[64]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.layer1(x) (line 8)
        self.assertEqual(str(id2type[66]), "class ResNet")	# Name self (line 8)
        self.assertEqual(str(id2type[69]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 8)
        self.assertEqual(str(id2type[71]), "NoneType")	# Assign
        self.assertEqual(str(id2type[72]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 9)
        self.assertEqual(str(id2type[74]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.layer2(x) (line 9)
        self.assertEqual(str(id2type[76]), "class ResNet")	# Name self (line 9)
        self.assertEqual(str(id2type[79]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 9)
        self.assertEqual(str(id2type[81]), "NoneType")	# Assign
        self.assertEqual(str(id2type[82]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 10)
        self.assertEqual(str(id2type[84]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.layer3(x) (line 10)
        self.assertEqual(str(id2type[86]), "class ResNet")	# Name self (line 10)
        self.assertEqual(str(id2type[89]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 10)
        self.assertEqual(str(id2type[91]), "NoneType")	# Assign
        self.assertEqual(str(id2type[92]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 11)
        self.assertEqual(str(id2type[94]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.layer4(x) (line 11)
        self.assertEqual(str(id2type[96]), "class ResNet")	# Name self (line 11)
        self.assertEqual(str(id2type[99]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 11)
        self.assertEqual(str(id2type[101]), "NoneType")	# Assign
        self.assertEqual(str(id2type[102]), "torch.Tensor(float32, (1, 512, 1, 1))")	# Name x (line 13)
        self.assertEqual(str(id2type[104]), "torch.Tensor(float32, (1, 512, 1, 1))")	# Call self.avgpool(x) (line 13)
        self.assertEqual(str(id2type[106]), "class ResNet")	# Name self (line 13)
        self.assertEqual(str(id2type[109]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 13)
        self.assertEqual(str(id2type[111]), "NoneType")	# Assign
        self.assertEqual(str(id2type[112]), "torch.Tensor(float32, (1, 512))")	# Name x (line 15)
        self.assertEqual(str(id2type[114]), "torch.Tensor(float32, (1, 512))")	# Call torch.flatten(x, start_dim=1) (line 15)
        self.assertEqual(str(id2type[119]), "torch.Tensor(float32, (1, 512, 1, 1))")	# Name x (line 15)
        self.assertEqual(str(id2type[122]), "int")	# Constant 1 (line 15)
        self.assertEqual(str(id2type[123]), "NoneType")	# Assign
        self.assertEqual(str(id2type[124]), "torch.Tensor(float32, (1, 1000))")	# Name x (line 16)
        self.assertEqual(str(id2type[126]), "torch.Tensor(float32, (1, 1000))")	# Call self.fc(x) (line 16)
        self.assertEqual(str(id2type[128]), "class ResNet")	# Name self (line 16)
        self.assertEqual(str(id2type[131]), "torch.Tensor(float32, (1, 512))")	# Name x (line 16)
        self.assertEqual(str(id2type[133]), "torch.Tensor(float32, (1, 1000))")	# Return
        self.assertEqual(str(id2type[134]), "torch.Tensor(float32, (1, 1000))")	# Name x (line 18)
        self.assertEqual(str(id2type[136]), "class BasicBlock -> torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[142]), "NoneType")	# Assign
        self.assertEqual(str(id2type[143]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name identity (line 2)
        self.assertEqual(str(id2type[145]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 2)
        self.assertEqual(str(id2type[147]), "NoneType")	# Assign
        self.assertEqual(str(id2type[148]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 4)
        self.assertEqual(str(id2type[150]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[152]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[155]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 4)
        self.assertEqual(str(id2type[157]), "NoneType")	# Assign
        self.assertEqual(str(id2type[158]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 5)
        self.assertEqual(str(id2type[160]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[162]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[165]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 5)
        self.assertEqual(str(id2type[167]), "NoneType")	# Assign
        self.assertEqual(str(id2type[168]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 6)
        self.assertEqual(str(id2type[170]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[172]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[175]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 6)
        self.assertEqual(str(id2type[177]), "NoneType")	# Assign
        self.assertEqual(str(id2type[178]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 8)
        self.assertEqual(str(id2type[180]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[182]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[185]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 8)
        self.assertEqual(str(id2type[187]), "NoneType")	# Assign
        self.assertEqual(str(id2type[188]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 9)
        self.assertEqual(str(id2type[190]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[192]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[195]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 9)
        self.assertEqual(str(id2type[197]), "NoneType")	# If
        self.assertEqual(str(id2type[198]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[199]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[200]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[204]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[205]), "NoneType")	# Assign
        self.assertEqual(str(id2type[206]), "a54 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[208]), "a54 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[210]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[213]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 12)
        self.assertEqual(str(id2type[215]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[216]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 14)
        self.assertEqual(str(id2type[219]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name identity (line 14)
        self.assertEqual(str(id2type[221]), "NoneType")	# Assign
        self.assertEqual(str(id2type[222]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 15)
        self.assertEqual(str(id2type[224]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[226]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[229]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 15)
        self.assertEqual(str(id2type[231]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Return
        self.assertEqual(str(id2type[232]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 17)
        self.assertEqual(str(id2type[234]), "class BasicBlock -> torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[240]), "NoneType")	# Assign
        self.assertEqual(str(id2type[241]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name identity (line 2)
        self.assertEqual(str(id2type[243]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 2)
        self.assertEqual(str(id2type[245]), "NoneType")	# Assign
        self.assertEqual(str(id2type[246]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 4)
        self.assertEqual(str(id2type[248]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[250]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[253]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 4)
        self.assertEqual(str(id2type[255]), "NoneType")	# Assign
        self.assertEqual(str(id2type[256]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 5)
        self.assertEqual(str(id2type[258]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[260]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[263]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 5)
        self.assertEqual(str(id2type[265]), "NoneType")	# Assign
        self.assertEqual(str(id2type[266]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 6)
        self.assertEqual(str(id2type[268]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[270]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[273]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 6)
        self.assertEqual(str(id2type[275]), "NoneType")	# Assign
        self.assertEqual(str(id2type[276]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 8)
        self.assertEqual(str(id2type[278]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[280]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[283]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 8)
        self.assertEqual(str(id2type[285]), "NoneType")	# Assign
        self.assertEqual(str(id2type[286]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 9)
        self.assertEqual(str(id2type[288]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[290]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[293]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 9)
        self.assertEqual(str(id2type[295]), "NoneType")	# If
        self.assertEqual(str(id2type[296]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[297]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[298]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[302]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[303]), "NoneType")	# Assign
        self.assertEqual(str(id2type[304]), "a76 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[306]), "a76 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[308]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[311]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 12)
        self.assertEqual(str(id2type[313]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[314]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 14)
        self.assertEqual(str(id2type[317]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name identity (line 14)
        self.assertEqual(str(id2type[319]), "NoneType")	# Assign
        self.assertEqual(str(id2type[320]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 15)
        self.assertEqual(str(id2type[322]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[324]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[327]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 15)
        self.assertEqual(str(id2type[329]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Return
        self.assertEqual(str(id2type[330]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 17)
        self.assertEqual(str(id2type[332]), "class BasicBlock -> torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[338]), "NoneType")	# Assign
        self.assertEqual(str(id2type[339]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name identity (line 2)
        self.assertEqual(str(id2type[341]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 2)
        self.assertEqual(str(id2type[343]), "NoneType")	# Assign
        self.assertEqual(str(id2type[344]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 4)
        self.assertEqual(str(id2type[346]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[348]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[351]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 4)
        self.assertEqual(str(id2type[353]), "NoneType")	# Assign
        self.assertEqual(str(id2type[354]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 5)
        self.assertEqual(str(id2type[356]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[358]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[361]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 5)
        self.assertEqual(str(id2type[363]), "NoneType")	# Assign
        self.assertEqual(str(id2type[364]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 6)
        self.assertEqual(str(id2type[366]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[368]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[371]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 6)
        self.assertEqual(str(id2type[373]), "NoneType")	# Assign
        self.assertEqual(str(id2type[374]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 8)
        self.assertEqual(str(id2type[376]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[378]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[381]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 8)
        self.assertEqual(str(id2type[383]), "NoneType")	# Assign
        self.assertEqual(str(id2type[384]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 9)
        self.assertEqual(str(id2type[386]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[388]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[391]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 9)
        self.assertEqual(str(id2type[393]), "NoneType")	# If
        self.assertEqual(str(id2type[394]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[395]), "class Sequential")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[396]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[400]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[401]), "NoneType")	# Assign
        self.assertEqual(str(id2type[402]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name identity (line 12)
        self.assertEqual(str(id2type[404]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[406]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[409]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 12)
        self.assertEqual(str(id2type[411]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[412]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 14)
        self.assertEqual(str(id2type[415]), "torch.Tensor(float32, (1, None, None, None))")	# Name identity (line 14)
        self.assertEqual(str(id2type[417]), "NoneType")	# Assign
        self.assertEqual(str(id2type[418]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 15)
        self.assertEqual(str(id2type[420]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[422]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[425]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 15)
        self.assertEqual(str(id2type[427]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Return
        self.assertEqual(str(id2type[428]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 17)
        self.assertEqual(str(id2type[430]), "class BasicBlock -> torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[436]), "NoneType")	# Assign
        self.assertEqual(str(id2type[437]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name identity (line 2)
        self.assertEqual(str(id2type[439]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 2)
        self.assertEqual(str(id2type[441]), "NoneType")	# Assign
        self.assertEqual(str(id2type[442]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 4)
        self.assertEqual(str(id2type[444]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[446]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[449]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 4)
        self.assertEqual(str(id2type[451]), "NoneType")	# Assign
        self.assertEqual(str(id2type[452]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 5)
        self.assertEqual(str(id2type[454]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[456]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[459]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 5)
        self.assertEqual(str(id2type[461]), "NoneType")	# Assign
        self.assertEqual(str(id2type[462]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 6)
        self.assertEqual(str(id2type[464]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[466]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[469]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 6)
        self.assertEqual(str(id2type[471]), "NoneType")	# Assign
        self.assertEqual(str(id2type[472]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 8)
        self.assertEqual(str(id2type[474]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[476]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[479]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 8)
        self.assertEqual(str(id2type[481]), "NoneType")	# Assign
        self.assertEqual(str(id2type[482]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 9)
        self.assertEqual(str(id2type[484]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[486]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[489]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 9)
        self.assertEqual(str(id2type[491]), "NoneType")	# If
        self.assertEqual(str(id2type[492]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[493]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[494]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[498]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[499]), "NoneType")	# Assign
        self.assertEqual(str(id2type[500]), "a120 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[502]), "a120 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[504]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[507]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 12)
        self.assertEqual(str(id2type[509]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[510]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 14)
        self.assertEqual(str(id2type[513]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name identity (line 14)
        self.assertEqual(str(id2type[515]), "NoneType")	# Assign
        self.assertEqual(str(id2type[516]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 15)
        self.assertEqual(str(id2type[518]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[520]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[523]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 15)
        self.assertEqual(str(id2type[525]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Return
        self.assertEqual(str(id2type[526]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 17)
        self.assertEqual(str(id2type[528]), "class BasicBlock -> torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[534]), "NoneType")	# Assign
        self.assertEqual(str(id2type[535]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name identity (line 2)
        self.assertEqual(str(id2type[537]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 2)
        self.assertEqual(str(id2type[539]), "NoneType")	# Assign
        self.assertEqual(str(id2type[540]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 4)
        self.assertEqual(str(id2type[542]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[544]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[547]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 4)
        self.assertEqual(str(id2type[549]), "NoneType")	# Assign
        self.assertEqual(str(id2type[550]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 5)
        self.assertEqual(str(id2type[552]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[554]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[557]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 5)
        self.assertEqual(str(id2type[559]), "NoneType")	# Assign
        self.assertEqual(str(id2type[560]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 6)
        self.assertEqual(str(id2type[562]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[564]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[567]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 6)
        self.assertEqual(str(id2type[569]), "NoneType")	# Assign
        self.assertEqual(str(id2type[570]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 8)
        self.assertEqual(str(id2type[572]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[574]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[577]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 8)
        self.assertEqual(str(id2type[579]), "NoneType")	# Assign
        self.assertEqual(str(id2type[580]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 9)
        self.assertEqual(str(id2type[582]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[584]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[587]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 9)
        self.assertEqual(str(id2type[589]), "NoneType")	# If
        self.assertEqual(str(id2type[590]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[591]), "class Sequential")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[592]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[596]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[597]), "NoneType")	# Assign
        self.assertEqual(str(id2type[598]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name identity (line 12)
        self.assertEqual(str(id2type[600]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[602]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[605]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 12)
        self.assertEqual(str(id2type[607]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[608]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 14)
        self.assertEqual(str(id2type[611]), "torch.Tensor(float32, (1, None, None, None))")	# Name identity (line 14)
        self.assertEqual(str(id2type[613]), "NoneType")	# Assign
        self.assertEqual(str(id2type[614]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 15)
        self.assertEqual(str(id2type[616]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[618]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[621]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 15)
        self.assertEqual(str(id2type[623]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Return
        self.assertEqual(str(id2type[624]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 17)
        self.assertEqual(str(id2type[626]), "class BasicBlock -> torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[632]), "NoneType")	# Assign
        self.assertEqual(str(id2type[633]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name identity (line 2)
        self.assertEqual(str(id2type[635]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 2)
        self.assertEqual(str(id2type[637]), "NoneType")	# Assign
        self.assertEqual(str(id2type[638]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 4)
        self.assertEqual(str(id2type[640]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[642]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[645]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 4)
        self.assertEqual(str(id2type[647]), "NoneType")	# Assign
        self.assertEqual(str(id2type[648]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 5)
        self.assertEqual(str(id2type[650]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[652]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[655]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 5)
        self.assertEqual(str(id2type[657]), "NoneType")	# Assign
        self.assertEqual(str(id2type[658]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 6)
        self.assertEqual(str(id2type[660]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[662]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[665]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 6)
        self.assertEqual(str(id2type[667]), "NoneType")	# Assign
        self.assertEqual(str(id2type[668]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 8)
        self.assertEqual(str(id2type[670]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[672]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[675]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 8)
        self.assertEqual(str(id2type[677]), "NoneType")	# Assign
        self.assertEqual(str(id2type[678]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 9)
        self.assertEqual(str(id2type[680]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[682]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[685]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 9)
        self.assertEqual(str(id2type[687]), "NoneType")	# If
        self.assertEqual(str(id2type[688]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[689]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[690]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[694]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[695]), "NoneType")	# Assign
        self.assertEqual(str(id2type[696]), "a164 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[698]), "a164 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[700]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[703]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 12)
        self.assertEqual(str(id2type[705]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[706]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 14)
        self.assertEqual(str(id2type[709]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name identity (line 14)
        self.assertEqual(str(id2type[711]), "NoneType")	# Assign
        self.assertEqual(str(id2type[712]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 15)
        self.assertEqual(str(id2type[714]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[716]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[719]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 15)
        self.assertEqual(str(id2type[721]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Return
        self.assertEqual(str(id2type[722]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 17)
        self.assertEqual(str(id2type[724]), "class BasicBlock -> torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[730]), "NoneType")	# Assign
        self.assertEqual(str(id2type[731]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name identity (line 2)
        self.assertEqual(str(id2type[733]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 2)
        self.assertEqual(str(id2type[735]), "NoneType")	# Assign
        self.assertEqual(str(id2type[736]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 4)
        self.assertEqual(str(id2type[738]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[740]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[743]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 4)
        self.assertEqual(str(id2type[745]), "NoneType")	# Assign
        self.assertEqual(str(id2type[746]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 5)
        self.assertEqual(str(id2type[748]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[750]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[753]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 5)
        self.assertEqual(str(id2type[755]), "NoneType")	# Assign
        self.assertEqual(str(id2type[756]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 6)
        self.assertEqual(str(id2type[758]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[760]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[763]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 6)
        self.assertEqual(str(id2type[765]), "NoneType")	# Assign
        self.assertEqual(str(id2type[766]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 8)
        self.assertEqual(str(id2type[768]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[770]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[773]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 8)
        self.assertEqual(str(id2type[775]), "NoneType")	# Assign
        self.assertEqual(str(id2type[776]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 9)
        self.assertEqual(str(id2type[778]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[780]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[783]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 9)
        self.assertEqual(str(id2type[785]), "NoneType")	# If
        self.assertEqual(str(id2type[786]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[787]), "class Sequential")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[788]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[792]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[793]), "NoneType")	# Assign
        self.assertEqual(str(id2type[794]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name identity (line 12)
        self.assertEqual(str(id2type[796]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[798]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[801]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 12)
        self.assertEqual(str(id2type[803]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[804]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 14)
        self.assertEqual(str(id2type[807]), "torch.Tensor(float32, (1, None, None, None))")	# Name identity (line 14)
        self.assertEqual(str(id2type[809]), "NoneType")	# Assign
        self.assertEqual(str(id2type[810]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 15)
        self.assertEqual(str(id2type[812]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[814]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[817]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 15)
        self.assertEqual(str(id2type[819]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Return
        self.assertEqual(str(id2type[820]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 17)
        self.assertEqual(str(id2type[822]), "class BasicBlock -> torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[828]), "NoneType")	# Assign
        self.assertEqual(str(id2type[829]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name identity (line 2)
        self.assertEqual(str(id2type[831]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 2)
        self.assertEqual(str(id2type[833]), "NoneType")	# Assign
        self.assertEqual(str(id2type[834]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 4)
        self.assertEqual(str(id2type[836]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[838]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[841]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 4)
        self.assertEqual(str(id2type[843]), "NoneType")	# Assign
        self.assertEqual(str(id2type[844]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 5)
        self.assertEqual(str(id2type[846]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[848]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[851]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 5)
        self.assertEqual(str(id2type[853]), "NoneType")	# Assign
        self.assertEqual(str(id2type[854]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 6)
        self.assertEqual(str(id2type[856]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[858]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[861]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 6)
        self.assertEqual(str(id2type[863]), "NoneType")	# Assign
        self.assertEqual(str(id2type[864]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 8)
        self.assertEqual(str(id2type[866]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[868]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[871]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 8)
        self.assertEqual(str(id2type[873]), "NoneType")	# Assign
        self.assertEqual(str(id2type[874]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 9)
        self.assertEqual(str(id2type[876]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[878]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[881]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 9)
        self.assertEqual(str(id2type[883]), "NoneType")	# If
        self.assertEqual(str(id2type[884]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[885]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[886]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[890]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[891]), "NoneType")	# Assign
        self.assertEqual(str(id2type[892]), "a208 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[894]), "a208 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[896]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[899]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 12)
        self.assertEqual(str(id2type[901]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[902]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 14)
        self.assertEqual(str(id2type[905]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name identity (line 14)
        self.assertEqual(str(id2type[907]), "NoneType")	# Assign
        self.assertEqual(str(id2type[908]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 15)
        self.assertEqual(str(id2type[910]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[912]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[915]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 15)
        self.assertEqual(str(id2type[917]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Return
        self.assertEqual(str(id2type[918]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 17)
        # === END ASSERTIONS for ResNet18 ===


def main():
    unittest.main()

if __name__ == '__main__':
    main()
