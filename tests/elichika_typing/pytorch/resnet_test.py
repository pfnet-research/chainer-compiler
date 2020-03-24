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
        self.assertEqual(str(id2type[9]), "torch.Tensor(float32, (1, 3, 224, 224)) -> torch.Tensor(float32, (1, 1000))")	# Attribute self._forward_impl (line 2)
        self.assertEqual(str(id2type[10]), "class ResNet")	# Name self (line 2)
        self.assertEqual(str(id2type[13]), "torch.Tensor(float32, (1, 3, 224, 224))")	# Name x (line 2)
        self.assertEqual(str(id2type[15]), "class ResNet -> torch.Tensor(float32, (1, 3, 224, 224)) -> torch.Tensor(float32, (1, 1000))")	# FunctionDef _forward_impl (line 1)
        self.assertEqual(str(id2type[21]), "NoneType")	# Assign
        self.assertEqual(str(id2type[22]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 3)
        self.assertEqual(str(id2type[24]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Call self.conv1(x) (line 3)
        self.assertEqual(str(id2type[25]), "torch.Tensor(float32, (1, 3, 224, 224)) -> torch.Tensor(float32, (1, 64, 112, 112))")	# Attribute self.conv1 (line 3)
        self.assertEqual(str(id2type[26]), "class ResNet")	# Name self (line 3)
        self.assertEqual(str(id2type[29]), "torch.Tensor(float32, (1, 3, 224, 224))")	# Name x (line 3)
        self.assertEqual(str(id2type[31]), "NoneType")	# Assign
        self.assertEqual(str(id2type[32]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 4)
        self.assertEqual(str(id2type[34]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Call self.bn1(x) (line 4)
        self.assertEqual(str(id2type[35]), "torch.Tensor(float32, (1, 64, 112, 112)) -> torch.Tensor(float32, (1, 64, 112, 112))")	# Attribute self.bn1 (line 4)
        self.assertEqual(str(id2type[36]), "class ResNet")	# Name self (line 4)
        self.assertEqual(str(id2type[39]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 4)
        self.assertEqual(str(id2type[41]), "NoneType")	# Assign
        self.assertEqual(str(id2type[42]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 5)
        self.assertEqual(str(id2type[44]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Call self.relu(x) (line 5)
        self.assertEqual(str(id2type[45]), "torch.Tensor(float32, (1, 64, 112, 112)) -> torch.Tensor(float32, (1, 64, 112, 112))")	# Attribute self.relu (line 5)
        self.assertEqual(str(id2type[46]), "class ResNet")	# Name self (line 5)
        self.assertEqual(str(id2type[49]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 5)
        self.assertEqual(str(id2type[51]), "NoneType")	# Assign
        self.assertEqual(str(id2type[52]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 6)
        self.assertEqual(str(id2type[54]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.maxpool(x) (line 6)
        self.assertEqual(str(id2type[55]), "torch.Tensor(float32, (1, 64, 112, 112)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.maxpool (line 6)
        self.assertEqual(str(id2type[56]), "class ResNet")	# Name self (line 6)
        self.assertEqual(str(id2type[59]), "torch.Tensor(float32, (1, 64, 112, 112))")	# Name x (line 6)
        self.assertEqual(str(id2type[61]), "NoneType")	# Assign
        self.assertEqual(str(id2type[62]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 8)
        self.assertEqual(str(id2type[64]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.layer1(x) (line 8)
        self.assertEqual(str(id2type[65]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.layer1 (line 8)
        self.assertEqual(str(id2type[66]), "class ResNet")	# Name self (line 8)
        self.assertEqual(str(id2type[69]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 8)
        self.assertEqual(str(id2type[71]), "NoneType")	# Assign
        self.assertEqual(str(id2type[72]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 9)
        self.assertEqual(str(id2type[74]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.layer2(x) (line 9)
        self.assertEqual(str(id2type[75]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Attribute self.layer2 (line 9)
        self.assertEqual(str(id2type[76]), "class ResNet")	# Name self (line 9)
        self.assertEqual(str(id2type[79]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 9)
        self.assertEqual(str(id2type[81]), "NoneType")	# Assign
        self.assertEqual(str(id2type[82]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 10)
        self.assertEqual(str(id2type[84]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.layer3(x) (line 10)
        self.assertEqual(str(id2type[85]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Attribute self.layer3 (line 10)
        self.assertEqual(str(id2type[86]), "class ResNet")	# Name self (line 10)
        self.assertEqual(str(id2type[89]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 10)
        self.assertEqual(str(id2type[91]), "NoneType")	# Assign
        self.assertEqual(str(id2type[92]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 11)
        self.assertEqual(str(id2type[94]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.layer4(x) (line 11)
        self.assertEqual(str(id2type[95]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Attribute self.layer4 (line 11)
        self.assertEqual(str(id2type[96]), "class ResNet")	# Name self (line 11)
        self.assertEqual(str(id2type[99]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 11)
        self.assertEqual(str(id2type[101]), "NoneType")	# Assign
        self.assertEqual(str(id2type[102]), "torch.Tensor(float32, (1, 512, 1, 1))")	# Name x (line 13)
        self.assertEqual(str(id2type[104]), "torch.Tensor(float32, (1, 512, 1, 1))")	# Call self.avgpool(x) (line 13)
        self.assertEqual(str(id2type[105]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 1, 1))")	# Attribute self.avgpool (line 13)
        self.assertEqual(str(id2type[106]), "class ResNet")	# Name self (line 13)
        self.assertEqual(str(id2type[109]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 13)
        self.assertEqual(str(id2type[111]), "NoneType")	# Assign
        self.assertEqual(str(id2type[112]), "torch.Tensor(float32, (1, 512))")	# Name x (line 15)
        self.assertEqual(str(id2type[114]), "torch.Tensor(float32, (1, 512))")	# Call torch.flatten(x, start_dim=1) (line 15)
        self.assertEqual(str(id2type[115]), "torch.Tensor(float32, (1, 512, 1, 1)) -> torch.Tensor(float32, (1, 512))")	# Attribute torch.flatten (line 15)
        self.assertEqual(str(id2type[119]), "torch.Tensor(float32, (1, 512, 1, 1))")	# Name x (line 15)
        self.assertEqual(str(id2type[122]), "int")	# Constant 1 (line 15)
        self.assertEqual(str(id2type[123]), "NoneType")	# Assign
        self.assertEqual(str(id2type[124]), "torch.Tensor(float32, (1, 1000))")	# Name x (line 16)
        self.assertEqual(str(id2type[126]), "torch.Tensor(float32, (1, 1000))")	# Call self.fc(x) (line 16)
        self.assertEqual(str(id2type[127]), "torch.Tensor(float32, (1, 512)) -> torch.Tensor(float32, (1, 1000))")	# Attribute self.fc (line 16)
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
        self.assertEqual(str(id2type[151]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.conv1 (line 4)
        self.assertEqual(str(id2type[152]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[155]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 4)
        self.assertEqual(str(id2type[157]), "NoneType")	# Assign
        self.assertEqual(str(id2type[158]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 5)
        self.assertEqual(str(id2type[160]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[161]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.bn1 (line 5)
        self.assertEqual(str(id2type[162]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[165]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 5)
        self.assertEqual(str(id2type[167]), "NoneType")	# Assign
        self.assertEqual(str(id2type[168]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 6)
        self.assertEqual(str(id2type[170]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[171]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.relu (line 6)
        self.assertEqual(str(id2type[172]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[175]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 6)
        self.assertEqual(str(id2type[177]), "NoneType")	# Assign
        self.assertEqual(str(id2type[178]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 8)
        self.assertEqual(str(id2type[180]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[181]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.conv2 (line 8)
        self.assertEqual(str(id2type[182]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[185]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 8)
        self.assertEqual(str(id2type[187]), "NoneType")	# Assign
        self.assertEqual(str(id2type[188]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 9)
        self.assertEqual(str(id2type[190]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[191]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.bn2 (line 9)
        self.assertEqual(str(id2type[192]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[195]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 9)
        self.assertEqual(str(id2type[197]), "NoneType")	# If
        self.assertEqual(str(id2type[198]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[199]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[200]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[204]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[205]), "NoneType")	# Assign
        self.assertEqual(str(id2type[206]), "a76 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[208]), "a76 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[209]), "torch.Tensor(float32, (1, 64, 56, 56)) -> a76 (from line 12)")	# Attribute self.downsample (line 12)
        self.assertEqual(str(id2type[210]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[213]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name x (line 12)
        self.assertEqual(str(id2type[215]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[216]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 14)
        self.assertEqual(str(id2type[218]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Add
        self.assertEqual(str(id2type[219]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name identity (line 14)
        self.assertEqual(str(id2type[221]), "NoneType")	# Assign
        self.assertEqual(str(id2type[222]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 15)
        self.assertEqual(str(id2type[224]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[225]), "torch.Tensor(float32, (1, 64, 56, 56)) -> torch.Tensor(float32, (1, 64, 56, 56))")	# Attribute self.relu (line 15)
        self.assertEqual(str(id2type[226]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[229]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 15)
        self.assertEqual(str(id2type[231]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Return
        self.assertEqual(str(id2type[232]), "torch.Tensor(float32, (1, 64, 56, 56))")	# Name out (line 17)
        self.assertEqual(str(id2type[234]), "class BasicBlock -> torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[240]), "NoneType")	# Assign
        self.assertEqual(str(id2type[241]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name identity (line 2)
        self.assertEqual(str(id2type[243]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 2)
        self.assertEqual(str(id2type[245]), "NoneType")	# Assign
        self.assertEqual(str(id2type[246]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 4)
        self.assertEqual(str(id2type[248]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[249]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Attribute self.conv1 (line 4)
        self.assertEqual(str(id2type[250]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[253]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 4)
        self.assertEqual(str(id2type[255]), "NoneType")	# Assign
        self.assertEqual(str(id2type[256]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 5)
        self.assertEqual(str(id2type[258]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[259]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Attribute self.bn1 (line 5)
        self.assertEqual(str(id2type[260]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[263]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 5)
        self.assertEqual(str(id2type[265]), "NoneType")	# Assign
        self.assertEqual(str(id2type[266]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 6)
        self.assertEqual(str(id2type[268]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[269]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Attribute self.relu (line 6)
        self.assertEqual(str(id2type[270]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[273]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 6)
        self.assertEqual(str(id2type[275]), "NoneType")	# Assign
        self.assertEqual(str(id2type[276]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 8)
        self.assertEqual(str(id2type[278]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[279]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Attribute self.conv2 (line 8)
        self.assertEqual(str(id2type[280]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[283]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 8)
        self.assertEqual(str(id2type[285]), "NoneType")	# Assign
        self.assertEqual(str(id2type[286]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 9)
        self.assertEqual(str(id2type[288]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[289]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Attribute self.bn2 (line 9)
        self.assertEqual(str(id2type[290]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[293]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 9)
        self.assertEqual(str(id2type[295]), "NoneType")	# If
        self.assertEqual(str(id2type[296]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[297]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[298]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[302]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[303]), "NoneType")	# Assign
        self.assertEqual(str(id2type[304]), "a120 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[306]), "a120 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[307]), "torch.Tensor(float32, (1, 128, 28, 28)) -> a120 (from line 12)")	# Attribute self.downsample (line 12)
        self.assertEqual(str(id2type[308]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[311]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name x (line 12)
        self.assertEqual(str(id2type[313]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[314]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 14)
        self.assertEqual(str(id2type[316]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Add
        self.assertEqual(str(id2type[317]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name identity (line 14)
        self.assertEqual(str(id2type[319]), "NoneType")	# Assign
        self.assertEqual(str(id2type[320]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 15)
        self.assertEqual(str(id2type[322]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[323]), "torch.Tensor(float32, (1, 128, 28, 28)) -> torch.Tensor(float32, (1, 128, 28, 28))")	# Attribute self.relu (line 15)
        self.assertEqual(str(id2type[324]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[327]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 15)
        self.assertEqual(str(id2type[329]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Return
        self.assertEqual(str(id2type[330]), "torch.Tensor(float32, (1, 128, 28, 28))")	# Name out (line 17)
        self.assertEqual(str(id2type[332]), "class BasicBlock -> torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[338]), "NoneType")	# Assign
        self.assertEqual(str(id2type[339]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name identity (line 2)
        self.assertEqual(str(id2type[341]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 2)
        self.assertEqual(str(id2type[343]), "NoneType")	# Assign
        self.assertEqual(str(id2type[344]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 4)
        self.assertEqual(str(id2type[346]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[347]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Attribute self.conv1 (line 4)
        self.assertEqual(str(id2type[348]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[351]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 4)
        self.assertEqual(str(id2type[353]), "NoneType")	# Assign
        self.assertEqual(str(id2type[354]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 5)
        self.assertEqual(str(id2type[356]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[357]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Attribute self.bn1 (line 5)
        self.assertEqual(str(id2type[358]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[361]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 5)
        self.assertEqual(str(id2type[363]), "NoneType")	# Assign
        self.assertEqual(str(id2type[364]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 6)
        self.assertEqual(str(id2type[366]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[367]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Attribute self.relu (line 6)
        self.assertEqual(str(id2type[368]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[371]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 6)
        self.assertEqual(str(id2type[373]), "NoneType")	# Assign
        self.assertEqual(str(id2type[374]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 8)
        self.assertEqual(str(id2type[376]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[377]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Attribute self.conv2 (line 8)
        self.assertEqual(str(id2type[378]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[381]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 8)
        self.assertEqual(str(id2type[383]), "NoneType")	# Assign
        self.assertEqual(str(id2type[384]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 9)
        self.assertEqual(str(id2type[386]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[387]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Attribute self.bn2 (line 9)
        self.assertEqual(str(id2type[388]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[391]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 9)
        self.assertEqual(str(id2type[393]), "NoneType")	# If
        self.assertEqual(str(id2type[394]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[395]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[396]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[400]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[401]), "NoneType")	# Assign
        self.assertEqual(str(id2type[402]), "a164 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[404]), "a164 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[405]), "torch.Tensor(float32, (1, 256, 14, 14)) -> a164 (from line 12)")	# Attribute self.downsample (line 12)
        self.assertEqual(str(id2type[406]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[409]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name x (line 12)
        self.assertEqual(str(id2type[411]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[412]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 14)
        self.assertEqual(str(id2type[414]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Add
        self.assertEqual(str(id2type[415]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name identity (line 14)
        self.assertEqual(str(id2type[417]), "NoneType")	# Assign
        self.assertEqual(str(id2type[418]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 15)
        self.assertEqual(str(id2type[420]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[421]), "torch.Tensor(float32, (1, 256, 14, 14)) -> torch.Tensor(float32, (1, 256, 14, 14))")	# Attribute self.relu (line 15)
        self.assertEqual(str(id2type[422]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[425]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 15)
        self.assertEqual(str(id2type[427]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Return
        self.assertEqual(str(id2type[428]), "torch.Tensor(float32, (1, 256, 14, 14))")	# Name out (line 17)
        self.assertEqual(str(id2type[430]), "class BasicBlock -> torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[436]), "NoneType")	# Assign
        self.assertEqual(str(id2type[437]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name identity (line 2)
        self.assertEqual(str(id2type[439]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 2)
        self.assertEqual(str(id2type[441]), "NoneType")	# Assign
        self.assertEqual(str(id2type[442]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 4)
        self.assertEqual(str(id2type[444]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.conv1(x) (line 4)
        self.assertEqual(str(id2type[445]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Attribute self.conv1 (line 4)
        self.assertEqual(str(id2type[446]), "class BasicBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[449]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 4)
        self.assertEqual(str(id2type[451]), "NoneType")	# Assign
        self.assertEqual(str(id2type[452]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 5)
        self.assertEqual(str(id2type[454]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.bn1(out) (line 5)
        self.assertEqual(str(id2type[455]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Attribute self.bn1 (line 5)
        self.assertEqual(str(id2type[456]), "class BasicBlock")	# Name self (line 5)
        self.assertEqual(str(id2type[459]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 5)
        self.assertEqual(str(id2type[461]), "NoneType")	# Assign
        self.assertEqual(str(id2type[462]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 6)
        self.assertEqual(str(id2type[464]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.relu(out) (line 6)
        self.assertEqual(str(id2type[465]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Attribute self.relu (line 6)
        self.assertEqual(str(id2type[466]), "class BasicBlock")	# Name self (line 6)
        self.assertEqual(str(id2type[469]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 6)
        self.assertEqual(str(id2type[471]), "NoneType")	# Assign
        self.assertEqual(str(id2type[472]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 8)
        self.assertEqual(str(id2type[474]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.conv2(out) (line 8)
        self.assertEqual(str(id2type[475]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Attribute self.conv2 (line 8)
        self.assertEqual(str(id2type[476]), "class BasicBlock")	# Name self (line 8)
        self.assertEqual(str(id2type[479]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 8)
        self.assertEqual(str(id2type[481]), "NoneType")	# Assign
        self.assertEqual(str(id2type[482]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 9)
        self.assertEqual(str(id2type[484]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.bn2(out) (line 9)
        self.assertEqual(str(id2type[485]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Attribute self.bn2 (line 9)
        self.assertEqual(str(id2type[486]), "class BasicBlock")	# Name self (line 9)
        self.assertEqual(str(id2type[489]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 9)
        self.assertEqual(str(id2type[491]), "NoneType")	# If
        self.assertEqual(str(id2type[492]), "bool")	# Compare  (line 11)
        self.assertEqual(str(id2type[493]), "NoneType")	# Attribute self.downsample (line 11)
        self.assertEqual(str(id2type[494]), "class BasicBlock")	# Name self (line 11)
        self.assertEqual(str(id2type[498]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[499]), "NoneType")	# Assign
        self.assertEqual(str(id2type[500]), "a208 (from line 12)")	# Name identity (line 12)
        self.assertEqual(str(id2type[502]), "a208 (from line 12)")	# Call self.downsample(x) (line 12)
        self.assertEqual(str(id2type[503]), "torch.Tensor(float32, (1, 512, 7, 7)) -> a208 (from line 12)")	# Attribute self.downsample (line 12)
        self.assertEqual(str(id2type[504]), "class BasicBlock")	# Name self (line 12)
        self.assertEqual(str(id2type[507]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name x (line 12)
        self.assertEqual(str(id2type[509]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[510]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 14)
        self.assertEqual(str(id2type[512]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Add
        self.assertEqual(str(id2type[513]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name identity (line 14)
        self.assertEqual(str(id2type[515]), "NoneType")	# Assign
        self.assertEqual(str(id2type[516]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 15)
        self.assertEqual(str(id2type[518]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Call self.relu(out) (line 15)
        self.assertEqual(str(id2type[519]), "torch.Tensor(float32, (1, 512, 7, 7)) -> torch.Tensor(float32, (1, 512, 7, 7))")	# Attribute self.relu (line 15)
        self.assertEqual(str(id2type[520]), "class BasicBlock")	# Name self (line 15)
        self.assertEqual(str(id2type[523]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 15)
        self.assertEqual(str(id2type[525]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Return
        self.assertEqual(str(id2type[526]), "torch.Tensor(float32, (1, 512, 7, 7))")	# Name out (line 17)
        # === END ASSERTIONS for ResNet18 ===


def main():
    unittest.main()

if __name__ == '__main__':
    main()
