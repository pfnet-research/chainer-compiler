import torch
import torch.nn as nn
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.pytorch.time_sequence_prediction import gen_Sequence_model


class TestSequence(unittest.TestCase):
    def test_Sequence(self):
        type_inference_tools.reset_state()

        model, forward_args = gen_Sequence_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for Sequence ===
        self.assertEqual(str(id2type[1]), "class Sequence -> torch.Tensor(float32, (3, 4)) -> int -> torch.Tensor(float64, (3, None))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[9]), "NoneType")	# Assign
        self.assertEqual(str(id2type[10]), "torch.Tensor(float64, (3, 1)) list")	# Name outputs (line 2)
        self.assertEqual(str(id2type[12]), "torch.Tensor(float64, (3, 1)) list")	# List [] (line 2)
        self.assertEqual(str(id2type[14]), "NoneType")	# Assign
        self.assertEqual(str(id2type[15]), "torch.Tensor(float64, (3, 51))")	# Name h_t (line 3)
        self.assertEqual(str(id2type[17]), "torch.Tensor(float64, (3, 51))")	# Call torch.zeros(input.size(0), 51, dtype=torch.double) (line 3)
        self.assertEqual(str(id2type[18]), "int -> int -> torch.Tensor(float64, (3, 51))")	# Attribute torch.zeros (line 3)
        self.assertEqual(str(id2type[22]), "int")	# Call input.size(0) (line 3)
        self.assertEqual(str(id2type[23]), "int -> int")	# Attribute input.size (line 3)
        self.assertEqual(str(id2type[24]), "torch.Tensor(float32, (3, 4))")	# Name input (line 3)
        self.assertEqual(str(id2type[27]), "int")	# Constant 0 (line 3)
        self.assertEqual(str(id2type[28]), "int")	# Constant 51 (line 3)
        self.assertEqual(str(id2type[30]), "dtype(float64)")	# Attribute torch.double (line 3)
        self.assertEqual(str(id2type[34]), "NoneType")	# Assign
        self.assertEqual(str(id2type[35]), "torch.Tensor(float64, (3, 51))")	# Name c_t (line 4)
        self.assertEqual(str(id2type[37]), "torch.Tensor(float64, (3, 51))")	# Call torch.zeros(input.size(0), 51, dtype=torch.double) (line 4)
        self.assertEqual(str(id2type[38]), "int -> int -> torch.Tensor(float64, (3, 51))")	# Attribute torch.zeros (line 4)
        self.assertEqual(str(id2type[42]), "int")	# Call input.size(0) (line 4)
        self.assertEqual(str(id2type[43]), "int -> int")	# Attribute input.size (line 4)
        self.assertEqual(str(id2type[44]), "torch.Tensor(float32, (3, 4))")	# Name input (line 4)
        self.assertEqual(str(id2type[47]), "int")	# Constant 0 (line 4)
        self.assertEqual(str(id2type[48]), "int")	# Constant 51 (line 4)
        self.assertEqual(str(id2type[50]), "dtype(float64)")	# Attribute torch.double (line 4)
        self.assertEqual(str(id2type[54]), "NoneType")	# Assign
        self.assertEqual(str(id2type[55]), "torch.Tensor(float64, (3, 51))")	# Name h_t2 (line 5)
        self.assertEqual(str(id2type[57]), "torch.Tensor(float64, (3, 51))")	# Call torch.zeros(input.size(0), 51, dtype=torch.double) (line 5)
        self.assertEqual(str(id2type[58]), "int -> int -> torch.Tensor(float64, (3, 51))")	# Attribute torch.zeros (line 5)
        self.assertEqual(str(id2type[62]), "int")	# Call input.size(0) (line 5)
        self.assertEqual(str(id2type[63]), "int -> int")	# Attribute input.size (line 5)
        self.assertEqual(str(id2type[64]), "torch.Tensor(float32, (3, 4))")	# Name input (line 5)
        self.assertEqual(str(id2type[67]), "int")	# Constant 0 (line 5)
        self.assertEqual(str(id2type[68]), "int")	# Constant 51 (line 5)
        self.assertEqual(str(id2type[70]), "dtype(float64)")	# Attribute torch.double (line 5)
        self.assertEqual(str(id2type[74]), "NoneType")	# Assign
        self.assertEqual(str(id2type[75]), "torch.Tensor(float64, (3, 51))")	# Name c_t2 (line 6)
        self.assertEqual(str(id2type[77]), "torch.Tensor(float64, (3, 51))")	# Call torch.zeros(input.size(0), 51, dtype=torch.double) (line 6)
        self.assertEqual(str(id2type[78]), "int -> int -> torch.Tensor(float64, (3, 51))")	# Attribute torch.zeros (line 6)
        self.assertEqual(str(id2type[82]), "int")	# Call input.size(0) (line 6)
        self.assertEqual(str(id2type[83]), "int -> int")	# Attribute input.size (line 6)
        self.assertEqual(str(id2type[84]), "torch.Tensor(float32, (3, 4))")	# Name input (line 6)
        self.assertEqual(str(id2type[87]), "int")	# Constant 0 (line 6)
        self.assertEqual(str(id2type[88]), "int")	# Constant 51 (line 6)
        self.assertEqual(str(id2type[90]), "dtype(float64)")	# Attribute torch.double (line 6)
        self.assertEqual(str(id2type[94]), "NoneType")	# For
        self.assertEqual(str(id2type[95]), "(int, torch.Tensor(float32, (3, 1)))")	# Tuple (i, input_t) (line 8)
        self.assertEqual(str(id2type[96]), "int")	# Name i (line 8)
        self.assertEqual(str(id2type[98]), "torch.Tensor(float32, (3, 1))")	# Name input_t (line 8)
        self.assertEqual(str(id2type[101]), "(int, torch.Tensor(float32, (3, 1))) list")	# Call enumerate(input.chunk(input.size(1), dim=1)) (line 8)
        self.assertEqual(str(id2type[102]), "(torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1))) -> (int, torch.Tensor(float32, (3, 1))) list")	# Name enumerate (line 8)
        self.assertEqual(str(id2type[104]), "(torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)))")	# Call input.chunk(input.size(1), dim=1) (line 8)
        self.assertEqual(str(id2type[105]), "int -> (torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)), torch.Tensor(float32, (3, 1)))")	# Attribute input.chunk (line 8)
        self.assertEqual(str(id2type[106]), "torch.Tensor(float32, (3, 4))")	# Name input (line 8)
        self.assertEqual(str(id2type[109]), "int")	# Call input.size(1) (line 8)
        self.assertEqual(str(id2type[110]), "int -> int")	# Attribute input.size (line 8)
        self.assertEqual(str(id2type[111]), "torch.Tensor(float32, (3, 4))")	# Name input (line 8)
        self.assertEqual(str(id2type[114]), "int")	# Constant 1 (line 8)
        self.assertEqual(str(id2type[116]), "int")	# Constant 1 (line 8)
        self.assertEqual(str(id2type[117]), "NoneType")	# Assign
        self.assertEqual(str(id2type[118]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t, c_t) (line 9)
        self.assertEqual(str(id2type[119]), "torch.Tensor(float64, (3, 51))")	# Name h_t (line 9)
        self.assertEqual(str(id2type[121]), "torch.Tensor(float64, (3, 51))")	# Name c_t (line 9)
        self.assertEqual(str(id2type[124]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Call self.lstm1(input_t, (h_t, c_t)) (line 9)
        self.assertEqual(str(id2type[125]), "torch.Tensor(float32, (3, 1)) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51))) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Attribute self.lstm1 (line 9)
        self.assertEqual(str(id2type[126]), "class Sequence")	# Name self (line 9)
        self.assertEqual(str(id2type[129]), "torch.Tensor(float32, (3, 1))")	# Name input_t (line 9)
        self.assertEqual(str(id2type[131]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t, c_t) (line 9)
        self.assertEqual(str(id2type[132]), "torch.Tensor(float64, (3, 51))")	# Name h_t (line 9)
        self.assertEqual(str(id2type[134]), "torch.Tensor(float64, (3, 51))")	# Name c_t (line 9)
        self.assertEqual(str(id2type[137]), "NoneType")	# Assign
        self.assertEqual(str(id2type[138]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t2, c_t2) (line 10)
        self.assertEqual(str(id2type[139]), "torch.Tensor(float64, (3, 51))")	# Name h_t2 (line 10)
        self.assertEqual(str(id2type[141]), "torch.Tensor(float64, (3, 51))")	# Name c_t2 (line 10)
        self.assertEqual(str(id2type[144]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Call self.lstm2(h_t, (h_t2, c_t2)) (line 10)
        self.assertEqual(str(id2type[145]), "torch.Tensor(float64, (3, 51)) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51))) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Attribute self.lstm2 (line 10)
        self.assertEqual(str(id2type[146]), "class Sequence")	# Name self (line 10)
        self.assertEqual(str(id2type[149]), "torch.Tensor(float64, (3, 51))")	# Name h_t (line 10)
        self.assertEqual(str(id2type[151]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t2, c_t2) (line 10)
        self.assertEqual(str(id2type[152]), "torch.Tensor(float64, (3, 51))")	# Name h_t2 (line 10)
        self.assertEqual(str(id2type[154]), "torch.Tensor(float64, (3, 51))")	# Name c_t2 (line 10)
        self.assertEqual(str(id2type[157]), "NoneType")	# Assign
        self.assertEqual(str(id2type[158]), "torch.Tensor(float64, (3, 1))")	# Name output (line 11)
        self.assertEqual(str(id2type[160]), "torch.Tensor(float64, (3, 1))")	# Call self.linear(h_t2) (line 11)
        self.assertEqual(str(id2type[161]), "torch.Tensor(float64, (3, 51)) -> torch.Tensor(float64, (3, 1))")	# Attribute self.linear (line 11)
        self.assertEqual(str(id2type[162]), "class Sequence")	# Name self (line 11)
        self.assertEqual(str(id2type[165]), "torch.Tensor(float64, (3, 51))")	# Name h_t2 (line 11)
        self.assertEqual(str(id2type[167]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[168]), "torch.Tensor(float64, (3, 1)) list")	# Name outputs (line 12)
        self.assertEqual(str(id2type[170]), "torch.Tensor(float64, (3, 1)) list -> torch.Tensor(float64, (3, 1)) list -> torch.Tensor(float64, (3, 1)) list")	# Add
        self.assertEqual(str(id2type[171]), "torch.Tensor(float64, (3, 1)) list")	# List [output] (line 12)
        self.assertEqual(str(id2type[172]), "torch.Tensor(float64, (3, 1))")	# Name output (line 12)
        self.assertEqual(str(id2type[175]), "NoneType")	# For
        self.assertEqual(str(id2type[176]), "int")	# Name i (line 13)
        self.assertEqual(str(id2type[178]), "int list")	# Call range(future) (line 13)
        self.assertEqual(str(id2type[179]), "int -> int list")	# Name range (line 13)
        self.assertEqual(str(id2type[181]), "int")	# Name future (line 13)
        self.assertEqual(str(id2type[183]), "NoneType")	# Assign
        self.assertEqual(str(id2type[184]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t, c_t) (line 14)
        self.assertEqual(str(id2type[185]), "torch.Tensor(float64, (3, 51))")	# Name h_t (line 14)
        self.assertEqual(str(id2type[187]), "torch.Tensor(float64, (3, 51))")	# Name c_t (line 14)
        self.assertEqual(str(id2type[190]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Call self.lstm1(output, (h_t, c_t)) (line 14)
        self.assertEqual(str(id2type[191]), "torch.Tensor(float64, (3, 1)) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51))) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Attribute self.lstm1 (line 14)
        self.assertEqual(str(id2type[192]), "class Sequence")	# Name self (line 14)
        self.assertEqual(str(id2type[195]), "torch.Tensor(float64, (3, 1))")	# Name output (line 14)
        self.assertEqual(str(id2type[197]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t, c_t) (line 14)
        self.assertEqual(str(id2type[198]), "torch.Tensor(float64, (3, 51))")	# Name h_t (line 14)
        self.assertEqual(str(id2type[200]), "torch.Tensor(float64, (3, 51))")	# Name c_t (line 14)
        self.assertEqual(str(id2type[203]), "NoneType")	# Assign
        self.assertEqual(str(id2type[204]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t2, c_t2) (line 15)
        self.assertEqual(str(id2type[205]), "torch.Tensor(float64, (3, 51))")	# Name h_t2 (line 15)
        self.assertEqual(str(id2type[207]), "torch.Tensor(float64, (3, 51))")	# Name c_t2 (line 15)
        self.assertEqual(str(id2type[210]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Call self.lstm2(h_t, (h_t2, c_t2)) (line 15)
        self.assertEqual(str(id2type[211]), "torch.Tensor(float64, (3, 51)) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51))) -> (torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Attribute self.lstm2 (line 15)
        self.assertEqual(str(id2type[212]), "class Sequence")	# Name self (line 15)
        self.assertEqual(str(id2type[215]), "torch.Tensor(float64, (3, 51))")	# Name h_t (line 15)
        self.assertEqual(str(id2type[217]), "(torch.Tensor(float64, (3, 51)), torch.Tensor(float64, (3, 51)))")	# Tuple (h_t2, c_t2) (line 15)
        self.assertEqual(str(id2type[218]), "torch.Tensor(float64, (3, 51))")	# Name h_t2 (line 15)
        self.assertEqual(str(id2type[220]), "torch.Tensor(float64, (3, 51))")	# Name c_t2 (line 15)
        self.assertEqual(str(id2type[223]), "NoneType")	# Assign
        self.assertEqual(str(id2type[224]), "torch.Tensor(float64, (3, 1))")	# Name output (line 16)
        self.assertEqual(str(id2type[226]), "torch.Tensor(float64, (3, 1))")	# Call self.linear(h_t2) (line 16)
        self.assertEqual(str(id2type[227]), "torch.Tensor(float64, (3, 51)) -> torch.Tensor(float64, (3, 1))")	# Attribute self.linear (line 16)
        self.assertEqual(str(id2type[228]), "class Sequence")	# Name self (line 16)
        self.assertEqual(str(id2type[231]), "torch.Tensor(float64, (3, 51))")	# Name h_t2 (line 16)
        self.assertEqual(str(id2type[233]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[234]), "torch.Tensor(float64, (3, 1)) list")	# Name outputs (line 17)
        self.assertEqual(str(id2type[236]), "torch.Tensor(float64, (3, 1)) list -> torch.Tensor(float64, (3, 1)) list -> torch.Tensor(float64, (3, 1)) list")	# Add
        self.assertEqual(str(id2type[237]), "torch.Tensor(float64, (3, 1)) list")	# List [output] (line 17)
        self.assertEqual(str(id2type[238]), "torch.Tensor(float64, (3, 1))")	# Name output (line 17)
        self.assertEqual(str(id2type[241]), "NoneType")	# Assign
        self.assertEqual(str(id2type[242]), "torch.Tensor(float64, (3, None))")	# Name outputs (line 19)
        self.assertEqual(str(id2type[244]), "torch.Tensor(float64, (3, None))")	# Call torch.stack(outputs, dim=1).squeeze(dim=2) (line 19)
        self.assertEqual(str(id2type[245]), "(no argument) -> torch.Tensor(float64, (3, None))")	# Attribute torch.stack(outputs, dim=1).squeeze (line 19)
        self.assertEqual(str(id2type[246]), "torch.Tensor(float64, (3, None, 1))")	# Call torch.stack(outputs, dim=1) (line 19)
        self.assertEqual(str(id2type[247]), "torch.Tensor(float64, (3, 1)) list -> torch.Tensor(float64, (3, None, 1))")	# Attribute torch.stack (line 19)
        self.assertEqual(str(id2type[251]), "torch.Tensor(float64, (3, 1)) list")	# Name outputs (line 19)
        self.assertEqual(str(id2type[254]), "int")	# Constant 1 (line 19)
        self.assertEqual(str(id2type[257]), "int")	# Constant 2 (line 19)
        self.assertEqual(str(id2type[258]), "torch.Tensor(float64, (3, None))")	# Return
        self.assertEqual(str(id2type[259]), "torch.Tensor(float64, (3, None))")	# Name outputs (line 20)
        # === END ASSERTIONS for Sequence ===

def main():
    unittest.main()

if __name__ == '__main__':
    main()
