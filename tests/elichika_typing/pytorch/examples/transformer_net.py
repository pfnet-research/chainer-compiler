import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.examples.transformer_net import gen_TransformerNet_test


class TestTransformerNet(unittest.TestCase):
    def test_TransformerNet(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_TransformerNet_test()
        id2type = generate_id2type_from_forward(model, forward_args)

        self.assertEqual(str(id2type[1]), "class TransformerNet -> torch.Tensor(float32, (5, 3, 16, 16)) -> torch.Tensor(float32, (5, 3, None, None))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[7]), "NoneType")	# Assign
        self.assertEqual(str(id2type[8]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Name y (line 2)
        self.assertEqual(str(id2type[10]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Call self.relu(self.in1(self.conv1(X))) (line 2)
        self.assertEqual(str(id2type[11]), "torch.Tensor(float32, (5, 32, 16, 16)) -> torch.Tensor(float32, (5, 32, 16, 16))")	# Attribute self.relu (line 2)
        self.assertEqual(str(id2type[12]), "class TransformerNet")	# Name self (line 2)
        self.assertEqual(str(id2type[15]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Call self.in1(self.conv1(X)) (line 2)
        self.assertEqual(str(id2type[16]), "torch.Tensor(float32, (5, 32, 16, 16)) -> torch.Tensor(float32, (5, 32, 16, 16))")	# Attribute self.in1 (line 2)
        self.assertEqual(str(id2type[17]), "class TransformerNet")	# Name self (line 2)
        self.assertEqual(str(id2type[20]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Call self.conv1(X) (line 2)
        self.assertEqual(str(id2type[21]), "torch.Tensor(float32, (5, 3, 16, 16)) -> torch.Tensor(float32, (5, 32, 16, 16))")	# Attribute self.conv1 (line 2)
        self.assertEqual(str(id2type[22]), "class TransformerNet")	# Name self (line 2)
        self.assertEqual(str(id2type[25]), "torch.Tensor(float32, (5, 3, 16, 16))")	# Name X (line 2)
        self.assertEqual(str(id2type[27]), "NoneType")	# Assign
        self.assertEqual(str(id2type[28]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Name y (line 3)
        self.assertEqual(str(id2type[30]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Call self.relu(self.in2(self.conv2(y))) (line 3)
        self.assertEqual(str(id2type[31]), "torch.Tensor(float32, (5, 64, 8, 8)) -> torch.Tensor(float32, (5, 64, 8, 8))")	# Attribute self.relu (line 3)
        self.assertEqual(str(id2type[32]), "class TransformerNet")	# Name self (line 3)
        self.assertEqual(str(id2type[35]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Call self.in2(self.conv2(y)) (line 3)
        self.assertEqual(str(id2type[36]), "torch.Tensor(float32, (5, 64, 8, 8)) -> torch.Tensor(float32, (5, 64, 8, 8))")	# Attribute self.in2 (line 3)
        self.assertEqual(str(id2type[37]), "class TransformerNet")	# Name self (line 3)
        self.assertEqual(str(id2type[40]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Call self.conv2(y) (line 3)
        self.assertEqual(str(id2type[41]), "torch.Tensor(float32, (5, 32, 16, 16)) -> torch.Tensor(float32, (5, 64, 8, 8))")	# Attribute self.conv2 (line 3)
        self.assertEqual(str(id2type[42]), "class TransformerNet")	# Name self (line 3)
        self.assertEqual(str(id2type[45]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Name y (line 3)
        self.assertEqual(str(id2type[47]), "NoneType")	# Assign
        self.assertEqual(str(id2type[48]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 4)
        self.assertEqual(str(id2type[50]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.relu(self.in3(self.conv3(y))) (line 4)
        self.assertEqual(str(id2type[51]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.relu (line 4)
        self.assertEqual(str(id2type[52]), "class TransformerNet")	# Name self (line 4)
        self.assertEqual(str(id2type[55]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in3(self.conv3(y)) (line 4)
        self.assertEqual(str(id2type[56]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in3 (line 4)
        self.assertEqual(str(id2type[57]), "class TransformerNet")	# Name self (line 4)
        self.assertEqual(str(id2type[60]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv3(y) (line 4)
        self.assertEqual(str(id2type[61]), "torch.Tensor(float32, (5, 64, 8, 8)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv3 (line 4)
        self.assertEqual(str(id2type[62]), "class TransformerNet")	# Name self (line 4)
        self.assertEqual(str(id2type[65]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Name y (line 4)
        self.assertEqual(str(id2type[67]), "NoneType")	# Assign
        self.assertEqual(str(id2type[68]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 5)
        self.assertEqual(str(id2type[70]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.res1(y) (line 5)
        self.assertEqual(str(id2type[71]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.res1 (line 5)
        self.assertEqual(str(id2type[72]), "class TransformerNet")	# Name self (line 5)
        self.assertEqual(str(id2type[75]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 5)
        self.assertEqual(str(id2type[77]), "NoneType")	# Assign
        self.assertEqual(str(id2type[78]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 6)
        self.assertEqual(str(id2type[80]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.res2(y) (line 6)
        self.assertEqual(str(id2type[81]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.res2 (line 6)
        self.assertEqual(str(id2type[82]), "class TransformerNet")	# Name self (line 6)
        self.assertEqual(str(id2type[85]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 6)
        self.assertEqual(str(id2type[87]), "NoneType")	# Assign
        self.assertEqual(str(id2type[88]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 7)
        self.assertEqual(str(id2type[90]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.res3(y) (line 7)
        self.assertEqual(str(id2type[91]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.res3 (line 7)
        self.assertEqual(str(id2type[92]), "class TransformerNet")	# Name self (line 7)
        self.assertEqual(str(id2type[95]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 7)
        self.assertEqual(str(id2type[97]), "NoneType")	# Assign
        self.assertEqual(str(id2type[98]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 8)
        self.assertEqual(str(id2type[100]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.res4(y) (line 8)
        self.assertEqual(str(id2type[101]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.res4 (line 8)
        self.assertEqual(str(id2type[102]), "class TransformerNet")	# Name self (line 8)
        self.assertEqual(str(id2type[105]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 8)
        self.assertEqual(str(id2type[107]), "NoneType")	# Assign
        self.assertEqual(str(id2type[108]), "torch.Tensor(float32, (5, 128, None, None))")	# Name y (line 9)
        self.assertEqual(str(id2type[110]), "torch.Tensor(float32, (5, 128, None, None))")	# Call self.res5(y) (line 9)
        self.assertEqual(str(id2type[111]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, None, None))")	# Attribute self.res5 (line 9)
        self.assertEqual(str(id2type[112]), "class TransformerNet")	# Name self (line 9)
        self.assertEqual(str(id2type[115]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name y (line 9)
        self.assertEqual(str(id2type[117]), "NoneType")	# Assign
        self.assertEqual(str(id2type[118]), "torch.Tensor(float32, (5, 64, None, None))")	# Name y (line 10)
        self.assertEqual(str(id2type[120]), "torch.Tensor(float32, (5, 64, None, None))")	# Call self.relu(self.in4(self.deconv1(y))) (line 10)
        self.assertEqual(str(id2type[121]), "torch.Tensor(float32, (5, 64, None, None)) -> torch.Tensor(float32, (5, 64, None, None))")	# Attribute self.relu (line 10)
        self.assertEqual(str(id2type[122]), "class TransformerNet")	# Name self (line 10)
        self.assertEqual(str(id2type[125]), "torch.Tensor(float32, (5, 64, None, None))")	# Call self.in4(self.deconv1(y)) (line 10)
        self.assertEqual(str(id2type[126]), "torch.Tensor(float32, (5, 64, None, None)) -> torch.Tensor(float32, (5, 64, None, None))")	# Attribute self.in4 (line 10)
        self.assertEqual(str(id2type[127]), "class TransformerNet")	# Name self (line 10)
        self.assertEqual(str(id2type[130]), "torch.Tensor(float32, (5, 64, None, None))")	# Call self.deconv1(y) (line 10)
        self.assertEqual(str(id2type[131]), "torch.Tensor(float32, (5, 128, None, None)) -> torch.Tensor(float32, (5, 64, None, None))")	# Attribute self.deconv1 (line 10)
        self.assertEqual(str(id2type[132]), "class TransformerNet")	# Name self (line 10)
        self.assertEqual(str(id2type[135]), "torch.Tensor(float32, (5, 128, None, None))")	# Name y (line 10)
        self.assertEqual(str(id2type[137]), "NoneType")	# Assign
        self.assertEqual(str(id2type[138]), "torch.Tensor(float32, (5, 32, None, None))")	# Name y (line 11)
        self.assertEqual(str(id2type[140]), "torch.Tensor(float32, (5, 32, None, None))")	# Call self.relu(self.in5(self.deconv2(y))) (line 11)
        self.assertEqual(str(id2type[141]), "torch.Tensor(float32, (5, 32, None, None)) -> torch.Tensor(float32, (5, 32, None, None))")	# Attribute self.relu (line 11)
        self.assertEqual(str(id2type[142]), "class TransformerNet")	# Name self (line 11)
        self.assertEqual(str(id2type[145]), "torch.Tensor(float32, (5, 32, None, None))")	# Call self.in5(self.deconv2(y)) (line 11)
        self.assertEqual(str(id2type[146]), "torch.Tensor(float32, (5, 32, None, None)) -> torch.Tensor(float32, (5, 32, None, None))")	# Attribute self.in5 (line 11)
        self.assertEqual(str(id2type[147]), "class TransformerNet")	# Name self (line 11)
        self.assertEqual(str(id2type[150]), "torch.Tensor(float32, (5, 32, None, None))")	# Call self.deconv2(y) (line 11)
        self.assertEqual(str(id2type[151]), "torch.Tensor(float32, (5, 64, None, None)) -> torch.Tensor(float32, (5, 32, None, None))")	# Attribute self.deconv2 (line 11)
        self.assertEqual(str(id2type[152]), "class TransformerNet")	# Name self (line 11)
        self.assertEqual(str(id2type[155]), "torch.Tensor(float32, (5, 64, None, None))")	# Name y (line 11)
        self.assertEqual(str(id2type[157]), "NoneType")	# Assign
        self.assertEqual(str(id2type[158]), "torch.Tensor(float32, (5, 3, None, None))")	# Name y (line 12)
        self.assertEqual(str(id2type[160]), "torch.Tensor(float32, (5, 3, None, None))")	# Call self.deconv3(y) (line 12)
        self.assertEqual(str(id2type[161]), "torch.Tensor(float32, (5, 32, None, None)) -> torch.Tensor(float32, (5, 3, None, None))")	# Attribute self.deconv3 (line 12)
        self.assertEqual(str(id2type[162]), "class TransformerNet")	# Name self (line 12)
        self.assertEqual(str(id2type[165]), "torch.Tensor(float32, (5, 32, None, None))")	# Name y (line 12)
        self.assertEqual(str(id2type[167]), "torch.Tensor(float32, (5, 3, None, None))")	# Return
        self.assertEqual(str(id2type[168]), "torch.Tensor(float32, (5, 3, None, None))")	# Name y (line 13)
        self.assertEqual(str(id2type[170]), "class ConvLayer -> torch.Tensor(float32, (5, 3, 16, 16)) -> torch.Tensor(float32, (5, 32, 16, 16))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[176]), "NoneType")	# Assign
        self.assertEqual(str(id2type[177]), "torch.Tensor(float32, (5, 3, 24, 24))")	# Name out (line 2)
        self.assertEqual(str(id2type[179]), "torch.Tensor(float32, (5, 3, 24, 24))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[180]), "torch.Tensor(float32, (5, 3, 16, 16)) -> torch.Tensor(float32, (5, 3, 24, 24))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[181]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[184]), "torch.Tensor(float32, (5, 3, 16, 16))")	# Name x (line 2)
        self.assertEqual(str(id2type[186]), "NoneType")	# Assign
        self.assertEqual(str(id2type[187]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Name out (line 3)
        self.assertEqual(str(id2type[189]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[190]), "torch.Tensor(float32, (5, 3, 24, 24)) -> torch.Tensor(float32, (5, 32, 16, 16))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[191]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[194]), "torch.Tensor(float32, (5, 3, 24, 24))")	# Name out (line 3)
        self.assertEqual(str(id2type[196]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Return
        self.assertEqual(str(id2type[197]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Name out (line 4)
        self.assertEqual(str(id2type[199]), "class ConvLayer -> torch.Tensor(float32, (5, 32, 16, 16)) -> torch.Tensor(float32, (5, 64, 8, 8))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[205]), "NoneType")	# Assign
        self.assertEqual(str(id2type[206]), "torch.Tensor(float32, (5, 32, 18, 18))")	# Name out (line 2)
        self.assertEqual(str(id2type[208]), "torch.Tensor(float32, (5, 32, 18, 18))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[209]), "torch.Tensor(float32, (5, 32, 16, 16)) -> torch.Tensor(float32, (5, 32, 18, 18))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[210]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[213]), "torch.Tensor(float32, (5, 32, 16, 16))")	# Name x (line 2)
        self.assertEqual(str(id2type[215]), "NoneType")	# Assign
        self.assertEqual(str(id2type[216]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Name out (line 3)
        self.assertEqual(str(id2type[218]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[219]), "torch.Tensor(float32, (5, 32, 18, 18)) -> torch.Tensor(float32, (5, 64, 8, 8))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[220]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[223]), "torch.Tensor(float32, (5, 32, 18, 18))")	# Name out (line 3)
        self.assertEqual(str(id2type[225]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Return
        self.assertEqual(str(id2type[226]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Name out (line 4)
        self.assertEqual(str(id2type[228]), "class ConvLayer -> torch.Tensor(float32, (5, 64, 8, 8)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[234]), "NoneType")	# Assign
        self.assertEqual(str(id2type[235]), "torch.Tensor(float32, (5, 64, 10, 10))")	# Name out (line 2)
        self.assertEqual(str(id2type[237]), "torch.Tensor(float32, (5, 64, 10, 10))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[238]), "torch.Tensor(float32, (5, 64, 8, 8)) -> torch.Tensor(float32, (5, 64, 10, 10))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[239]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[242]), "torch.Tensor(float32, (5, 64, 8, 8))")	# Name x (line 2)
        self.assertEqual(str(id2type[244]), "NoneType")	# Assign
        self.assertEqual(str(id2type[245]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[247]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[248]), "torch.Tensor(float32, (5, 64, 10, 10)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[249]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[252]), "torch.Tensor(float32, (5, 64, 10, 10))")	# Name out (line 3)
        self.assertEqual(str(id2type[254]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[255]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[257]), "class ResidualBlock -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[263]), "NoneType")	# Assign
        self.assertEqual(str(id2type[264]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 2)
        self.assertEqual(str(id2type[266]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[268]), "NoneType")	# Assign
        self.assertEqual(str(id2type[269]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[271]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.relu(self.in1(self.conv1(x))) (line 3)
        self.assertEqual(str(id2type[272]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.relu (line 3)
        self.assertEqual(str(id2type[273]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[276]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in1(self.conv1(x)) (line 3)
        self.assertEqual(str(id2type[277]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in1 (line 3)
        self.assertEqual(str(id2type[278]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[281]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv1(x) (line 3)
        self.assertEqual(str(id2type[282]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv1 (line 3)
        self.assertEqual(str(id2type[283]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[286]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 3)
        self.assertEqual(str(id2type[288]), "NoneType")	# Assign
        self.assertEqual(str(id2type[289]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[291]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in2(self.conv2(out)) (line 4)
        self.assertEqual(str(id2type[292]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in2 (line 4)
        self.assertEqual(str(id2type[293]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[296]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2(out) (line 4)
        self.assertEqual(str(id2type[297]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2 (line 4)
        self.assertEqual(str(id2type[298]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[301]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[303]), "NoneType")	# Assign
        self.assertEqual(str(id2type[304]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[306]), "torch.Tensor(float32, (5, 128, 4, 4))")	# BinOp out + residual (line 5)
        self.assertEqual(str(id2type[307]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[309]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Add
        self.assertEqual(str(id2type[310]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 5)
        self.assertEqual(str(id2type[312]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[313]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 6)
        self.assertEqual(str(id2type[315]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[321]), "NoneType")	# Assign
        self.assertEqual(str(id2type[322]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[324]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[325]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[326]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[329]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[331]), "NoneType")	# Assign
        self.assertEqual(str(id2type[332]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[334]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[335]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[336]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[339]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[341]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[342]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[344]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[350]), "NoneType")	# Assign
        self.assertEqual(str(id2type[351]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[353]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[354]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[355]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[358]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[360]), "NoneType")	# Assign
        self.assertEqual(str(id2type[361]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[363]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[364]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[365]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[368]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[370]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[371]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[373]), "class ResidualBlock -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[379]), "NoneType")	# Assign
        self.assertEqual(str(id2type[380]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 2)
        self.assertEqual(str(id2type[382]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[384]), "NoneType")	# Assign
        self.assertEqual(str(id2type[385]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[387]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.relu(self.in1(self.conv1(x))) (line 3)
        self.assertEqual(str(id2type[388]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.relu (line 3)
        self.assertEqual(str(id2type[389]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[392]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in1(self.conv1(x)) (line 3)
        self.assertEqual(str(id2type[393]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in1 (line 3)
        self.assertEqual(str(id2type[394]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[397]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv1(x) (line 3)
        self.assertEqual(str(id2type[398]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv1 (line 3)
        self.assertEqual(str(id2type[399]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[402]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 3)
        self.assertEqual(str(id2type[404]), "NoneType")	# Assign
        self.assertEqual(str(id2type[405]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[407]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in2(self.conv2(out)) (line 4)
        self.assertEqual(str(id2type[408]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in2 (line 4)
        self.assertEqual(str(id2type[409]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[412]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2(out) (line 4)
        self.assertEqual(str(id2type[413]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2 (line 4)
        self.assertEqual(str(id2type[414]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[417]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[419]), "NoneType")	# Assign
        self.assertEqual(str(id2type[420]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[422]), "torch.Tensor(float32, (5, 128, 4, 4))")	# BinOp out + residual (line 5)
        self.assertEqual(str(id2type[423]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[425]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Add
        self.assertEqual(str(id2type[426]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 5)
        self.assertEqual(str(id2type[428]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[429]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 6)
        self.assertEqual(str(id2type[431]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[437]), "NoneType")	# Assign
        self.assertEqual(str(id2type[438]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[440]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[441]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[442]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[445]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[447]), "NoneType")	# Assign
        self.assertEqual(str(id2type[448]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[450]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[451]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[452]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[455]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[457]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[458]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[460]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[466]), "NoneType")	# Assign
        self.assertEqual(str(id2type[467]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[469]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[470]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[471]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[474]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[476]), "NoneType")	# Assign
        self.assertEqual(str(id2type[477]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[479]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[480]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[481]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[484]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[486]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[487]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[489]), "class ResidualBlock -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[495]), "NoneType")	# Assign
        self.assertEqual(str(id2type[496]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 2)
        self.assertEqual(str(id2type[498]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[500]), "NoneType")	# Assign
        self.assertEqual(str(id2type[501]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[503]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.relu(self.in1(self.conv1(x))) (line 3)
        self.assertEqual(str(id2type[504]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.relu (line 3)
        self.assertEqual(str(id2type[505]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[508]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in1(self.conv1(x)) (line 3)
        self.assertEqual(str(id2type[509]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in1 (line 3)
        self.assertEqual(str(id2type[510]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[513]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv1(x) (line 3)
        self.assertEqual(str(id2type[514]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv1 (line 3)
        self.assertEqual(str(id2type[515]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[518]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 3)
        self.assertEqual(str(id2type[520]), "NoneType")	# Assign
        self.assertEqual(str(id2type[521]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[523]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in2(self.conv2(out)) (line 4)
        self.assertEqual(str(id2type[524]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in2 (line 4)
        self.assertEqual(str(id2type[525]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[528]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2(out) (line 4)
        self.assertEqual(str(id2type[529]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2 (line 4)
        self.assertEqual(str(id2type[530]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[533]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[535]), "NoneType")	# Assign
        self.assertEqual(str(id2type[536]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[538]), "torch.Tensor(float32, (5, 128, 4, 4))")	# BinOp out + residual (line 5)
        self.assertEqual(str(id2type[539]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[541]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Add
        self.assertEqual(str(id2type[542]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 5)
        self.assertEqual(str(id2type[544]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[545]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 6)
        self.assertEqual(str(id2type[547]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[553]), "NoneType")	# Assign
        self.assertEqual(str(id2type[554]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[556]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[557]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[558]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[561]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[563]), "NoneType")	# Assign
        self.assertEqual(str(id2type[564]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[566]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[567]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[568]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[571]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[573]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[574]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[576]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[582]), "NoneType")	# Assign
        self.assertEqual(str(id2type[583]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[585]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[586]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[587]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[590]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[592]), "NoneType")	# Assign
        self.assertEqual(str(id2type[593]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[595]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[596]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[597]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[600]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[602]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[603]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[605]), "class ResidualBlock -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[611]), "NoneType")	# Assign
        self.assertEqual(str(id2type[612]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 2)
        self.assertEqual(str(id2type[614]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[616]), "NoneType")	# Assign
        self.assertEqual(str(id2type[617]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[619]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.relu(self.in1(self.conv1(x))) (line 3)
        self.assertEqual(str(id2type[620]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.relu (line 3)
        self.assertEqual(str(id2type[621]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[624]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in1(self.conv1(x)) (line 3)
        self.assertEqual(str(id2type[625]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in1 (line 3)
        self.assertEqual(str(id2type[626]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[629]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv1(x) (line 3)
        self.assertEqual(str(id2type[630]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv1 (line 3)
        self.assertEqual(str(id2type[631]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[634]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 3)
        self.assertEqual(str(id2type[636]), "NoneType")	# Assign
        self.assertEqual(str(id2type[637]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[639]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in2(self.conv2(out)) (line 4)
        self.assertEqual(str(id2type[640]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in2 (line 4)
        self.assertEqual(str(id2type[641]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[644]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2(out) (line 4)
        self.assertEqual(str(id2type[645]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2 (line 4)
        self.assertEqual(str(id2type[646]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[649]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[651]), "NoneType")	# Assign
        self.assertEqual(str(id2type[652]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[654]), "torch.Tensor(float32, (5, 128, 4, 4))")	# BinOp out + residual (line 5)
        self.assertEqual(str(id2type[655]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[657]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Add
        self.assertEqual(str(id2type[658]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 5)
        self.assertEqual(str(id2type[660]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[661]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 6)
        self.assertEqual(str(id2type[663]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[669]), "NoneType")	# Assign
        self.assertEqual(str(id2type[670]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[672]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[673]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[674]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[677]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[679]), "NoneType")	# Assign
        self.assertEqual(str(id2type[680]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[682]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[683]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[684]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[687]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[689]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[690]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[692]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[698]), "NoneType")	# Assign
        self.assertEqual(str(id2type[699]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[701]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[702]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[703]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[706]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[708]), "NoneType")	# Assign
        self.assertEqual(str(id2type[709]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[711]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[712]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[713]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[716]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[718]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[719]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[721]), "class ResidualBlock -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, None, None))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[727]), "NoneType")	# Assign
        self.assertEqual(str(id2type[728]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 2)
        self.assertEqual(str(id2type[730]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[732]), "NoneType")	# Assign
        self.assertEqual(str(id2type[733]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[735]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.relu(self.in1(self.conv1(x))) (line 3)
        self.assertEqual(str(id2type[736]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.relu (line 3)
        self.assertEqual(str(id2type[737]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[740]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in1(self.conv1(x)) (line 3)
        self.assertEqual(str(id2type[741]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in1 (line 3)
        self.assertEqual(str(id2type[742]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[745]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv1(x) (line 3)
        self.assertEqual(str(id2type[746]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv1 (line 3)
        self.assertEqual(str(id2type[747]), "class ResidualBlock")	# Name self (line 3)
        self.assertEqual(str(id2type[750]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 3)
        self.assertEqual(str(id2type[752]), "NoneType")	# Assign
        self.assertEqual(str(id2type[753]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[755]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.in2(self.conv2(out)) (line 4)
        self.assertEqual(str(id2type[756]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.in2 (line 4)
        self.assertEqual(str(id2type[757]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[760]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2(out) (line 4)
        self.assertEqual(str(id2type[761]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2 (line 4)
        self.assertEqual(str(id2type[762]), "class ResidualBlock")	# Name self (line 4)
        self.assertEqual(str(id2type[765]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[767]), "NoneType")	# Assign
        self.assertEqual(str(id2type[768]), "torch.Tensor(float32, (5, 128, None, None))")	# Name out (line 5)
        self.assertEqual(str(id2type[770]), "torch.Tensor(float32, (5, 128, None, None))")	# BinOp out + residual (line 5)
        self.assertEqual(str(id2type[771]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 5)
        self.assertEqual(str(id2type[773]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, None, None))")	# Add
        self.assertEqual(str(id2type[774]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name residual (line 5)
        self.assertEqual(str(id2type[776]), "torch.Tensor(float32, (5, 128, None, None))")	# Return
        self.assertEqual(str(id2type[777]), "torch.Tensor(float32, (5, 128, None, None))")	# Name out (line 6)
        self.assertEqual(str(id2type[779]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[785]), "NoneType")	# Assign
        self.assertEqual(str(id2type[786]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[788]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[789]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[790]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[793]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[795]), "NoneType")	# Assign
        self.assertEqual(str(id2type[796]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[798]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[799]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[800]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[803]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[805]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[806]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[808]), "class ConvLayer -> torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[814]), "NoneType")	# Assign
        self.assertEqual(str(id2type[815]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 2)
        self.assertEqual(str(id2type[817]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[818]), "torch.Tensor(float32, (5, 128, 4, 4)) -> torch.Tensor(float32, (5, 128, 6, 6))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[819]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[822]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name x (line 2)
        self.assertEqual(str(id2type[824]), "NoneType")	# Assign
        self.assertEqual(str(id2type[825]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 3)
        self.assertEqual(str(id2type[827]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[828]), "torch.Tensor(float32, (5, 128, 6, 6)) -> torch.Tensor(float32, (5, 128, 4, 4))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[829]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[832]), "torch.Tensor(float32, (5, 128, 6, 6))")	# Name out (line 3)
        self.assertEqual(str(id2type[834]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Return
        self.assertEqual(str(id2type[835]), "torch.Tensor(float32, (5, 128, 4, 4))")	# Name out (line 4)
        self.assertEqual(str(id2type[837]), "class UpsampleConvLayer -> torch.Tensor(float32, (5, 128, None, None)) -> torch.Tensor(float32, (5, 64, None, None))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[843]), "NoneType")	# Assign
        self.assertEqual(str(id2type[844]), "torch.Tensor(float32, (5, 128, None, None))")	# Name x_in (line 2)
        self.assertEqual(str(id2type[846]), "torch.Tensor(float32, (5, 128, None, None))")	# Name x (line 2)
        self.assertEqual(str(id2type[848]), "NoneType")	# If
        self.assertEqual(str(id2type[849]), "int")	# Attribute self.upsample (line 3)
        self.assertEqual(str(id2type[850]), "class UpsampleConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[853]), "NoneType")	# Assign
        self.assertEqual(str(id2type[854]), "torch.Tensor(float32, (5, 128, None, None))")	# Name x_in (line 4)
        self.assertEqual(str(id2type[856]), "torch.Tensor(float32, (5, 128, None, None))")	# Call torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample) (line 4)
        self.assertEqual(str(id2type[857]), "torch.Tensor(float32, (5, 128, None, None)) -> torch.Tensor(float32, (5, 128, None, None))")	# Attribute torch.nn.functional.interpolate (line 4)
        self.assertEqual(str(id2type[858]), "class module")	# Attribute torch.nn.functional (line 4)
        self.assertEqual(str(id2type[859]), "class module")	# Attribute torch.nn (line 4)
        self.assertEqual(str(id2type[865]), "torch.Tensor(float32, (5, 128, None, None))")	# Name x_in (line 4)
        self.assertEqual(str(id2type[868]), "string")	# Constant 'nearest' (line 4)
        self.assertEqual(str(id2type[870]), "int")	# Attribute self.upsample (line 4)
        self.assertEqual(str(id2type[871]), "class UpsampleConvLayer")	# Name self (line 4)
        self.assertEqual(str(id2type[874]), "NoneType")	# Assign
        self.assertEqual(str(id2type[875]), "torch.Tensor(float32, (5, 128, None, None))")	# Name out (line 5)
        self.assertEqual(str(id2type[877]), "torch.Tensor(float32, (5, 128, None, None))")	# Call self.reflection_pad(x_in) (line 5)
        self.assertEqual(str(id2type[878]), "torch.Tensor(float32, (5, 128, None, None)) -> torch.Tensor(float32, (5, 128, None, None))")	# Attribute self.reflection_pad (line 5)
        self.assertEqual(str(id2type[879]), "class UpsampleConvLayer")	# Name self (line 5)
        self.assertEqual(str(id2type[882]), "torch.Tensor(float32, (5, 128, None, None))")	# Name x_in (line 5)
        self.assertEqual(str(id2type[884]), "NoneType")	# Assign
        self.assertEqual(str(id2type[885]), "torch.Tensor(float32, (5, 64, None, None))")	# Name out (line 6)
        self.assertEqual(str(id2type[887]), "torch.Tensor(float32, (5, 64, None, None))")	# Call self.conv2d(out) (line 6)
        self.assertEqual(str(id2type[888]), "torch.Tensor(float32, (5, 128, None, None)) -> torch.Tensor(float32, (5, 64, None, None))")	# Attribute self.conv2d (line 6)
        self.assertEqual(str(id2type[889]), "class UpsampleConvLayer")	# Name self (line 6)
        self.assertEqual(str(id2type[892]), "torch.Tensor(float32, (5, 128, None, None))")	# Name out (line 6)
        self.assertEqual(str(id2type[894]), "torch.Tensor(float32, (5, 64, None, None))")	# Return
        self.assertEqual(str(id2type[895]), "torch.Tensor(float32, (5, 64, None, None))")	# Name out (line 7)
        self.assertEqual(str(id2type[897]), "class UpsampleConvLayer -> torch.Tensor(float32, (5, 64, None, None)) -> torch.Tensor(float32, (5, 32, None, None))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[903]), "NoneType")	# Assign
        self.assertEqual(str(id2type[904]), "torch.Tensor(float32, (5, 64, None, None))")	# Name x_in (line 2)
        self.assertEqual(str(id2type[906]), "torch.Tensor(float32, (5, 64, None, None))")	# Name x (line 2)
        self.assertEqual(str(id2type[908]), "NoneType")	# If
        self.assertEqual(str(id2type[909]), "int")	# Attribute self.upsample (line 3)
        self.assertEqual(str(id2type[910]), "class UpsampleConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[913]), "NoneType")	# Assign
        self.assertEqual(str(id2type[914]), "torch.Tensor(float32, (5, 64, None, None))")	# Name x_in (line 4)
        self.assertEqual(str(id2type[916]), "torch.Tensor(float32, (5, 64, None, None))")	# Call torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample) (line 4)
        self.assertEqual(str(id2type[917]), "torch.Tensor(float32, (5, 64, None, None)) -> torch.Tensor(float32, (5, 64, None, None))")	# Attribute torch.nn.functional.interpolate (line 4)
        self.assertEqual(str(id2type[918]), "class module")	# Attribute torch.nn.functional (line 4)
        self.assertEqual(str(id2type[919]), "class module")	# Attribute torch.nn (line 4)
        self.assertEqual(str(id2type[925]), "torch.Tensor(float32, (5, 64, None, None))")	# Name x_in (line 4)
        self.assertEqual(str(id2type[928]), "string")	# Constant 'nearest' (line 4)
        self.assertEqual(str(id2type[930]), "int")	# Attribute self.upsample (line 4)
        self.assertEqual(str(id2type[931]), "class UpsampleConvLayer")	# Name self (line 4)
        self.assertEqual(str(id2type[934]), "NoneType")	# Assign
        self.assertEqual(str(id2type[935]), "torch.Tensor(float32, (5, 64, None, None))")	# Name out (line 5)
        self.assertEqual(str(id2type[937]), "torch.Tensor(float32, (5, 64, None, None))")	# Call self.reflection_pad(x_in) (line 5)
        self.assertEqual(str(id2type[938]), "torch.Tensor(float32, (5, 64, None, None)) -> torch.Tensor(float32, (5, 64, None, None))")	# Attribute self.reflection_pad (line 5)
        self.assertEqual(str(id2type[939]), "class UpsampleConvLayer")	# Name self (line 5)
        self.assertEqual(str(id2type[942]), "torch.Tensor(float32, (5, 64, None, None))")	# Name x_in (line 5)
        self.assertEqual(str(id2type[944]), "NoneType")	# Assign
        self.assertEqual(str(id2type[945]), "torch.Tensor(float32, (5, 32, None, None))")	# Name out (line 6)
        self.assertEqual(str(id2type[947]), "torch.Tensor(float32, (5, 32, None, None))")	# Call self.conv2d(out) (line 6)
        self.assertEqual(str(id2type[948]), "torch.Tensor(float32, (5, 64, None, None)) -> torch.Tensor(float32, (5, 32, None, None))")	# Attribute self.conv2d (line 6)
        self.assertEqual(str(id2type[949]), "class UpsampleConvLayer")	# Name self (line 6)
        self.assertEqual(str(id2type[952]), "torch.Tensor(float32, (5, 64, None, None))")	# Name out (line 6)
        self.assertEqual(str(id2type[954]), "torch.Tensor(float32, (5, 32, None, None))")	# Return
        self.assertEqual(str(id2type[955]), "torch.Tensor(float32, (5, 32, None, None))")	# Name out (line 7)
        self.assertEqual(str(id2type[957]), "class ConvLayer -> torch.Tensor(float32, (5, 32, None, None)) -> torch.Tensor(float32, (5, 3, None, None))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[963]), "NoneType")	# Assign
        self.assertEqual(str(id2type[964]), "torch.Tensor(float32, (5, 32, None, None))")	# Name out (line 2)
        self.assertEqual(str(id2type[966]), "torch.Tensor(float32, (5, 32, None, None))")	# Call self.reflection_pad(x) (line 2)
        self.assertEqual(str(id2type[967]), "torch.Tensor(float32, (5, 32, None, None)) -> torch.Tensor(float32, (5, 32, None, None))")	# Attribute self.reflection_pad (line 2)
        self.assertEqual(str(id2type[968]), "class ConvLayer")	# Name self (line 2)
        self.assertEqual(str(id2type[971]), "torch.Tensor(float32, (5, 32, None, None))")	# Name x (line 2)
        self.assertEqual(str(id2type[973]), "NoneType")	# Assign
        self.assertEqual(str(id2type[974]), "torch.Tensor(float32, (5, 3, None, None))")	# Name out (line 3)
        self.assertEqual(str(id2type[976]), "torch.Tensor(float32, (5, 3, None, None))")	# Call self.conv2d(out) (line 3)
        self.assertEqual(str(id2type[977]), "torch.Tensor(float32, (5, 32, None, None)) -> torch.Tensor(float32, (5, 3, None, None))")	# Attribute self.conv2d (line 3)
        self.assertEqual(str(id2type[978]), "class ConvLayer")	# Name self (line 3)
        self.assertEqual(str(id2type[981]), "torch.Tensor(float32, (5, 32, None, None))")	# Name out (line 3)
        self.assertEqual(str(id2type[983]), "torch.Tensor(float32, (5, 3, None, None))")	# Return
        self.assertEqual(str(id2type[984]), "torch.Tensor(float32, (5, 3, None, None))")	# Name out (line 4)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
