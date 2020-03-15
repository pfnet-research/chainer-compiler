import chainer
import numpy as np
import unittest

from chainer_compiler.elichika.testtools import generate_id2type_from_forward
from chainer_compiler.elichika.testtools import type_inference_tools

from testcases.elichika_tests.utils import sequence_utils
from testcases.elichika_tests.model.EspNet_AttDot import AttDot
from testcases.elichika_tests.model.EspNet_AttLoc import AttLoc
from testcases.elichika_tests.model.EspNet_BLSTM import BLSTM
from testcases.elichika_tests.model.EspNet_Decoder import Decoder
from testcases.elichika_tests.model.EspNet_E2E import E2E, test_recipe
from testcases.elichika_tests.model.EspNet_VGG2L import VGG2L
from testcases.elichika_tests.model.StatelessLSTM import StatelessLSTM

def gen_AttDot_model():
    type_inference_tools.reset_state()
    eprojs = 3
    dunits = 4
    att_dim = 5
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10

    model = AttDot(eprojs, dunits, att_dim)
    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, eprojs).astype(np.float32))
    forward_args = (xs, None, None)

    return model, forward_args


def gen_AttLoc_model():
    type_inference_tools.reset_state()
    eprojs = 3
    dunits = 4
    att_dim = 5
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10
    aconv_chans = 7
    aconv_filts = 6

    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, eprojs).astype(dtype=np.float32))
    model = AttLoc(eprojs, dunits, att_dim, aconv_chans, aconv_filts)
    forward_args = (xs, None, None)

    return model, forward_args


def gen_StatelessLSTM_model():
    type_inference_tools.reset_state()
    batch_size = 3
    in_size = 7
    out_size = 4

    c = chainer.Variable(np.random.rand(batch_size, out_size).astype(np.float32))
    h = chainer.Variable(np.random.rand(batch_size, out_size).astype(np.float32))
    x = chainer.Variable(np.random.rand(batch_size, in_size).astype(np.float32))

    model = StatelessLSTM(in_size, out_size)
    forward_args = (c, h, x)

    return model, forward_args


def gen_VGG2L_model():
    type_inference_tools.reset_state()
    idim = 5
    elayers = 2
    cdim = 3
    hdim = 7
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10

    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, idim).astype(dtype=np.float32))

    model = VGG2L(1)
    forward_args = (xs, ilens)

    return model, forward_args


def gen_BLSTM_model():
    type_inference_tools.reset_state()
    idim = 5
    elayers = 2
    cdim = 3
    hdim = 7
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10

    model = BLSTM(idim, elayers, cdim, hdim, 0)
    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, idim).astype(dtype=np.float32))

    forward_args = (xs, ilens)

    return model, forward_args


def gen_Decoder_model():
    type_inference_tools.reset_state()
    eprojs = 3
    dunits = 4
    att_dim = 5
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10
    dlayers = 2
    odim = 11
    sos = odim - 1
    eos = odim - 1
    aconv_chans = 7
    aconv_filts = 6

    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    hs = []
    for l in ilens:
        hs.append(np.random.rand(l, eprojs).astype(dtype=np.float32))

    ys, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, odim)

    model = Decoder(eprojs, odim, dlayers, dunits, sos, eos, att_dim)
    forward_args = (hs, ys)

    return model, forward_args


def gen_E2E_model():
    type_inference_tools.reset_state()
    (idim, odim, args), (xs, ilens, ys) = test_recipe()
    model = E2E(idim, odim, args, nobias=True)
    forward_args = (xs, ilens, ys)
    return model, forward_args


class TestEspNet(unittest.TestCase):
    def test_AttDot(self):
        model, forward_args = gen_AttDot_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for AttDot ===
        self.assertEqual(str(id2type[1]), "class AttDot -> [ndarray(float32, (4, 3)), ndarray(float32, (3, 3)), ndarray(float32, (3, 3))] -> NoneType -> NoneType -> (Variable(float32, (3, 3)), Variable(float32, (3, 4)))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[11]), "NoneType")	# Expr
        self.assertEqual(str(id2type[12]), "string")	# Constant "..." (line 8)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign
        self.assertEqual(str(id2type[14]), "float")	# Name scaling (line 10)
        self.assertEqual(str(id2type[16]), "float")	# Constant 2.0 (line 10)
        self.assertEqual(str(id2type[17]), "NoneType")	# Assign
        self.assertEqual(str(id2type[18]), "int")	# Name batch (line 11)
        self.assertEqual(str(id2type[20]), "int")	# Call len(enc_hs) (line 11)
        self.assertEqual(str(id2type[21]), "[ndarray(float32, (4, 3)), ndarray(float32, (3, 3)), ndarray(float32, (3, 3))] -> int")	# Name len (line 11)
        self.assertEqual(str(id2type[23]), "[ndarray(float32, (4, 3)), ndarray(float32, (3, 3)), ndarray(float32, (3, 3))]")	# Name enc_hs (line 11)
        self.assertEqual(str(id2type[25]), "NoneType")	# If
        self.assertEqual(str(id2type[26]), "bool")	# Compare  (line 14)
        self.assertEqual(str(id2type[27]), "NoneType")	# Attribute self.enc_h (line 14)
        self.assertEqual(str(id2type[28]), "class AttDot")	# Name self (line 14)
        self.assertEqual(str(id2type[32]), "NoneType")	# Constant None (line 14)
        self.assertEqual(str(id2type[33]), "NoneType")	# Assign
        self.assertEqual(str(id2type[34]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 15)
        self.assertEqual(str(id2type[35]), "class AttDot")	# Name self (line 15)
        self.assertEqual(str(id2type[38]), "Variable(float32, (3, 4, 3))")	# Call F.pad_sequence(enc_hs) (line 15)
        self.assertEqual(str(id2type[39]), "[ndarray(float32, (4, 3)), ndarray(float32, (3, 3)), ndarray(float32, (3, 3))] -> Variable(float32, (3, 4, 3))")	# Attribute F.pad_sequence (line 15)
        self.assertEqual(str(id2type[43]), "[ndarray(float32, (4, 3)), ndarray(float32, (3, 3)), ndarray(float32, (3, 3))]")	# Name enc_hs (line 15)
        self.assertEqual(str(id2type[45]), "NoneType")	# If
        self.assertEqual(str(id2type[46]), "bool")	# Compare  (line 17)
        self.assertEqual(str(id2type[47]), "NoneType")	# Attribute self.pre_compute_enc_h (line 17)
        self.assertEqual(str(id2type[48]), "class AttDot")	# Name self (line 17)
        self.assertEqual(str(id2type[52]), "NoneType")	# Constant None (line 17)
        self.assertEqual(str(id2type[53]), "NoneType")	# Assign
        self.assertEqual(str(id2type[54]), "int")	# Attribute self.h_length (line 18)
        self.assertEqual(str(id2type[55]), "class AttDot")	# Name self (line 18)
        self.assertEqual(str(id2type[58]), "int")	# Subscript self.enc_h.shape[1] (line 18)
        self.assertEqual(str(id2type[59]), "(int, int, int)")	# Attribute self.enc_h.shape (line 18)
        self.assertEqual(str(id2type[60]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 18)
        self.assertEqual(str(id2type[61]), "class AttDot")	# Name self (line 18)
        self.assertEqual(str(id2type[66]), "int")	# Constant 1 (line 18)
        self.assertEqual(str(id2type[68]), "NoneType")	# Assign
        self.assertEqual(str(id2type[69]), "Variable(float32, (3, 4, 5))")	# Attribute self.pre_compute_enc_h (line 20)
        self.assertEqual(str(id2type[70]), "class AttDot")	# Name self (line 20)
        self.assertEqual(str(id2type[73]), "Variable(float32, (3, 4, 5))")	# Call F.tanh(linear_tensor(self.mlp_enc, self.enc_h)) (line 20)
        self.assertEqual(str(id2type[74]), "Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5))")	# Attribute F.tanh (line 20)
        self.assertEqual(str(id2type[78]), "Variable(float32, (3, 4, 5))")	# Call linear_tensor(self.mlp_enc, self.enc_h) (line 21)
        self.assertEqual(str(id2type[79]), "class Linear -> Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 5))")	# Name linear_tensor (line 21)
        self.assertEqual(str(id2type[81]), "class Linear")	# Attribute self.mlp_enc (line 21)
        self.assertEqual(str(id2type[82]), "class AttDot")	# Name self (line 21)
        self.assertEqual(str(id2type[85]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 21)
        self.assertEqual(str(id2type[86]), "class AttDot")	# Name self (line 21)
        self.assertEqual(str(id2type[89]), "NoneType")	# If
        self.assertEqual(str(id2type[90]), "bool")	# Compare  (line 23)
        self.assertEqual(str(id2type[91]), "NoneType")	# Name dec_z (line 23)
        self.assertEqual(str(id2type[94]), "NoneType")	# Constant None (line 23)
        self.assertEqual(str(id2type[95]), "NoneType")	# Assign
        self.assertEqual(str(id2type[96]), "Variable(float32, (3, 4))")	# Name dec_z (line 24)
        self.assertEqual(str(id2type[98]), "Variable(float32, (3, 4))")	# Call chainer.Variable(self.xp.zeros((batch, self.dunits), dtype=np.float32)) (line 24)
        self.assertEqual(str(id2type[99]), "ndarray(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute chainer.Variable (line 24)
        self.assertEqual(str(id2type[103]), "ndarray(float32, (3, 4))")	# Call self.xp.zeros((batch, self.dunits), dtype=np.float32) (line 24)
        self.assertEqual(str(id2type[104]), "(int, int) -> ndarray(float32, (3, 4))")	# Attribute self.xp.zeros (line 24)
        self.assertEqual(str(id2type[105]), "class module")	# Attribute self.xp (line 24)
        self.assertEqual(str(id2type[106]), "class AttDot")	# Name self (line 24)
        self.assertEqual(str(id2type[110]), "(int, int)")	# Tuple (batch, self.dunits) (line 25)
        self.assertEqual(str(id2type[111]), "int")	# Name batch (line 25)
        self.assertEqual(str(id2type[113]), "int")	# Attribute self.dunits (line 25)
        self.assertEqual(str(id2type[114]), "class AttDot")	# Name self (line 25)
        self.assertEqual(str(id2type[119]), "dtype(float32)")	# Attribute np.float32 (line 25)
        self.assertEqual(str(id2type[123]), "NoneType")	# Assign
        self.assertEqual(str(id2type[124]), "a12")	# Name dec_z (line 27)
        self.assertEqual(str(id2type[126]), "a10 (from line 27)")	# Call F.reshape(dec_z, (batch, self.dunits)) (line 27)
        self.assertEqual(str(id2type[127]), "NoneType -> (int, int) -> a10 (from line 27)")	# Attribute F.reshape (line 27)
        self.assertEqual(str(id2type[131]), "NoneType")	# Name dec_z (line 27)
        self.assertEqual(str(id2type[133]), "(int, int)")	# Tuple (batch, self.dunits) (line 27)
        self.assertEqual(str(id2type[134]), "int")	# Name batch (line 27)
        self.assertEqual(str(id2type[136]), "int")	# Attribute self.dunits (line 27)
        self.assertEqual(str(id2type[137]), "class AttDot")	# Name self (line 27)
        self.assertEqual(str(id2type[141]), "NoneType")	# Assign
        self.assertEqual(str(id2type[142]), "Variable(float32, (3, 4, 5))")	# Name u (line 30)
        self.assertEqual(str(id2type[144]), "Variable(float32, (3, 4, 5))")	# Call F.broadcast_to(F.expand_dims(F.tanh(self.mlp_dec(dec_z)), 1), self.pre_compute_enc_h.shape) (line 30)
        self.assertEqual(str(id2type[145]), "Variable(float32, (3, 1, 5)) -> (int, int, int) -> Variable(float32, (3, 4, 5))")	# Attribute F.broadcast_to (line 30)
        self.assertEqual(str(id2type[149]), "Variable(float32, (3, 1, 5))")	# Call F.expand_dims(F.tanh(self.mlp_dec(dec_z)), 1) (line 30)
        self.assertEqual(str(id2type[150]), "Variable(float32, (3, 5)) -> int -> Variable(float32, (3, 1, 5))")	# Attribute F.expand_dims (line 30)
        self.assertEqual(str(id2type[154]), "Variable(float32, (3, 5))")	# Call F.tanh(self.mlp_dec(dec_z)) (line 30)
        self.assertEqual(str(id2type[155]), "Variable(float32, (3, 5)) -> Variable(float32, (3, 5))")	# Attribute F.tanh (line 30)
        self.assertEqual(str(id2type[159]), "Variable(float32, (3, 5))")	# Call self.mlp_dec(dec_z) (line 30)
        self.assertEqual(str(id2type[160]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 5))")	# Attribute self.mlp_dec (line 30)
        self.assertEqual(str(id2type[161]), "class AttDot")	# Name self (line 30)
        self.assertEqual(str(id2type[164]), "Variable(float32, (3, 4))")	# Name dec_z (line 30)
        self.assertEqual(str(id2type[166]), "int")	# Constant 1 (line 30)
        self.assertEqual(str(id2type[167]), "(int, int, int)")	# Attribute self.pre_compute_enc_h.shape (line 31)
        self.assertEqual(str(id2type[168]), "Variable(float32, (3, 4, 5))")	# Attribute self.pre_compute_enc_h (line 31)
        self.assertEqual(str(id2type[169]), "class AttDot")	# Name self (line 31)
        self.assertEqual(str(id2type[173]), "NoneType")	# Assign
        self.assertEqual(str(id2type[174]), "Variable(float32, (3, 4))")	# Name e (line 32)
        self.assertEqual(str(id2type[176]), "Variable(float32, (3, 4))")	# Call F.sum(self.pre_compute_enc_h * u, axis=2) (line 32)
        self.assertEqual(str(id2type[177]), "Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4))")	# Attribute F.sum (line 32)
        self.assertEqual(str(id2type[181]), "Variable(float32, (3, 4, 5))")	# BinOp self.pre_compute_enc_h * u (line 32)
        self.assertEqual(str(id2type[182]), "Variable(float32, (3, 4, 5))")	# Attribute self.pre_compute_enc_h (line 32)
        self.assertEqual(str(id2type[183]), "class AttDot")	# Name self (line 32)
        self.assertEqual(str(id2type[186]), "Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5))")	# Mult
        self.assertEqual(str(id2type[187]), "Variable(float32, (3, 4, 5))")	# Name u (line 32)
        self.assertEqual(str(id2type[190]), "int")	# Constant 2 (line 32)
        self.assertEqual(str(id2type[191]), "NoneType")	# Assign
        self.assertEqual(str(id2type[192]), "Variable(float32, (3, 4))")	# Name w (line 36)
        self.assertEqual(str(id2type[194]), "Variable(float32, (3, 4))")	# Call F.softmax(scaling * e) (line 36)
        self.assertEqual(str(id2type[195]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute F.softmax (line 36)
        self.assertEqual(str(id2type[199]), "Variable(float32, (3, 4))")	# BinOp scaling * e (line 36)
        self.assertEqual(str(id2type[200]), "float")	# Name scaling (line 36)
        self.assertEqual(str(id2type[202]), "float -> Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Mult
        self.assertEqual(str(id2type[203]), "Variable(float32, (3, 4))")	# Name e (line 36)
        self.assertEqual(str(id2type[205]), "NoneType")	# Assign
        self.assertEqual(str(id2type[206]), "Variable(float32, (3, 3))")	# Name c (line 39)
        self.assertEqual(str(id2type[208]), "Variable(float32, (3, 3))")	# Call F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1) (line 39)
        self.assertEqual(str(id2type[209]), "Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 3))")	# Attribute F.sum (line 39)
        self.assertEqual(str(id2type[213]), "Variable(float32, (3, 4, 3))")	# BinOp self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape) (line 39)
        self.assertEqual(str(id2type[214]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 39)
        self.assertEqual(str(id2type[215]), "class AttDot")	# Name self (line 39)
        self.assertEqual(str(id2type[218]), "Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 3))")	# Mult
        self.assertEqual(str(id2type[219]), "Variable(float32, (3, 4, 3))")	# Call F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape) (line 39)
        self.assertEqual(str(id2type[220]), "Variable(float32, (3, 4, 1)) -> (int, int, int) -> Variable(float32, (3, 4, 3))")	# Attribute F.broadcast_to (line 39)
        self.assertEqual(str(id2type[224]), "Variable(float32, (3, 4, 1))")	# Call F.expand_dims(w, 2) (line 39)
        self.assertEqual(str(id2type[225]), "Variable(float32, (3, 4)) -> int -> Variable(float32, (3, 4, 1))")	# Attribute F.expand_dims (line 39)
        self.assertEqual(str(id2type[229]), "Variable(float32, (3, 4))")	# Name w (line 39)
        self.assertEqual(str(id2type[231]), "int")	# Constant 2 (line 39)
        self.assertEqual(str(id2type[232]), "(int, int, int)")	# Attribute self.enc_h.shape (line 39)
        self.assertEqual(str(id2type[233]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 39)
        self.assertEqual(str(id2type[234]), "class AttDot")	# Name self (line 39)
        self.assertEqual(str(id2type[239]), "int")	# Constant 1 (line 39)
        self.assertEqual(str(id2type[240]), "(Variable(float32, (3, 3)), Variable(float32, (3, 4)))")	# Return
        self.assertEqual(str(id2type[241]), "(Variable(float32, (3, 3)), Variable(float32, (3, 4)))")	# Tuple (c, w) (line 41)
        self.assertEqual(str(id2type[242]), "Variable(float32, (3, 3))")	# Name c (line 41)
        self.assertEqual(str(id2type[244]), "Variable(float32, (3, 4))")	# Name w (line 41)
        self.assertEqual(str(id2type[247]), "class Linear -> Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 5))")	# FunctionDef linear_tensor (line 1)
        self.assertEqual(str(id2type[253]), "NoneType")	# Expr
        self.assertEqual(str(id2type[254]), "string")	# Constant "..." (line 8)
        self.assertEqual(str(id2type[255]), "NoneType")	# Assign
        self.assertEqual(str(id2type[256]), "Variable(float32, (12, 5))")	# Name y (line 9)
        self.assertEqual(str(id2type[258]), "Variable(float32, (12, 5))")	# Call linear(F.reshape(x, (-1, x.shape[-1]))) (line 9)
        self.assertEqual(str(id2type[259]), "Variable(float32, (12, 3)) -> Variable(float32, (12, 5))")	# Name linear (line 9)
        self.assertEqual(str(id2type[261]), "Variable(float32, (12, 3))")	# Call F.reshape(x, (-1, x.shape[-1])) (line 9)
        self.assertEqual(str(id2type[262]), "Variable(float32, (3, 4, 3)) -> (int, int) -> Variable(float32, (12, 3))")	# Attribute F.reshape (line 9)
        self.assertEqual(str(id2type[266]), "Variable(float32, (3, 4, 3))")	# Name x (line 9)
        self.assertEqual(str(id2type[268]), "(int, int)")	# Tuple (-1, x.shape[-1]) (line 9)
        self.assertEqual(str(id2type[269]), "int")	# UnaryOp -1 (line 9)
        self.assertEqual(str(id2type[271]), "int")	# Constant 1 (line 9)
        self.assertEqual(str(id2type[272]), "int")	# Subscript x.shape[-1] (line 9)
        self.assertEqual(str(id2type[273]), "(int, int, int)")	# Attribute x.shape (line 9)
        self.assertEqual(str(id2type[274]), "Variable(float32, (3, 4, 3))")	# Name x (line 9)
        self.assertEqual(str(id2type[278]), "int")	# UnaryOp -1 (line 9)
        self.assertEqual(str(id2type[280]), "int")	# Constant 1 (line 9)
        self.assertEqual(str(id2type[283]), "Variable(float32, (3, 4, 5))")	# Return
        self.assertEqual(str(id2type[284]), "Variable(float32, (3, 4, 5))")	# Call F.reshape(y, x.shape[:-1:] + (-1)) (line 10)
        self.assertEqual(str(id2type[285]), "Variable(float32, (12, 5)) -> (int, int, int) -> Variable(float32, (3, 4, 5))")	# Attribute F.reshape (line 10)
        self.assertEqual(str(id2type[289]), "Variable(float32, (12, 5))")	# Name y (line 10)
        self.assertEqual(str(id2type[291]), "(int, int, int)")	# BinOp x.shape[:-1:] + (-1) (line 10)
        self.assertEqual(str(id2type[292]), "(int, int)")	# Subscript x.shape[:-1:] (line 10)
        self.assertEqual(str(id2type[293]), "(int, int, int)")	# Attribute x.shape (line 10)
        self.assertEqual(str(id2type[294]), "Variable(float32, (3, 4, 3))")	# Name x (line 10)
        self.assertEqual(str(id2type[298]), "int")	# UnaryOp -1 (line 10)
        self.assertEqual(str(id2type[300]), "int")	# Constant 1 (line 10)
        self.assertEqual(str(id2type[302]), "(int, int) -> (int,) -> (int, int, int)")	# Add
        self.assertEqual(str(id2type[303]), "(int,)")	# Tuple (-1) (line 10)
        self.assertEqual(str(id2type[304]), "int")	# UnaryOp -1 (line 10)
        self.assertEqual(str(id2type[306]), "int")	# Constant 1 (line 10)
        # === END ASSERTIONS for AttDot ===


    def test_AttLoc(self):
        model, forward_args = gen_AttLoc_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for AttLoc ===
        self.assertEqual(str(id2type[1]), "class AttLoc -> [ndarray(float32, (4, 3)), ndarray(float32, (2, 3)), ndarray(float32, (2, 3))] -> NoneType -> NoneType -> (Variable(float32, (3, 3)), Variable(float32, (3, 4)))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[11]), "NoneType")	# Expr
        self.assertEqual(str(id2type[12]), "string")	# Constant "..." (line 9)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign
        self.assertEqual(str(id2type[14]), "float")	# Name scaling (line 11)
        self.assertEqual(str(id2type[16]), "float")	# Constant 2.0 (line 11)
        self.assertEqual(str(id2type[17]), "NoneType")	# Assign
        self.assertEqual(str(id2type[18]), "int")	# Name batch (line 12)
        self.assertEqual(str(id2type[20]), "int")	# Call len(enc_hs) (line 12)
        self.assertEqual(str(id2type[21]), "[ndarray(float32, (4, 3)), ndarray(float32, (2, 3)), ndarray(float32, (2, 3))] -> int")	# Name len (line 12)
        self.assertEqual(str(id2type[23]), "[ndarray(float32, (4, 3)), ndarray(float32, (2, 3)), ndarray(float32, (2, 3))]")	# Name enc_hs (line 12)
        self.assertEqual(str(id2type[25]), "NoneType")	# If
        self.assertEqual(str(id2type[26]), "bool")	# Compare  (line 15)
        self.assertEqual(str(id2type[27]), "NoneType")	# Attribute self.enc_h (line 15)
        self.assertEqual(str(id2type[28]), "class AttLoc")	# Name self (line 15)
        self.assertEqual(str(id2type[32]), "NoneType")	# Constant None (line 15)
        self.assertEqual(str(id2type[33]), "NoneType")	# Assign
        self.assertEqual(str(id2type[34]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 16)
        self.assertEqual(str(id2type[35]), "class AttLoc")	# Name self (line 16)
        self.assertEqual(str(id2type[38]), "Variable(float32, (3, 4, 3))")	# Call F.pad_sequence(enc_hs) (line 16)
        self.assertEqual(str(id2type[39]), "[ndarray(float32, (4, 3)), ndarray(float32, (2, 3)), ndarray(float32, (2, 3))] -> Variable(float32, (3, 4, 3))")	# Attribute F.pad_sequence (line 16)
        self.assertEqual(str(id2type[43]), "[ndarray(float32, (4, 3)), ndarray(float32, (2, 3)), ndarray(float32, (2, 3))]")	# Name enc_hs (line 16)
        self.assertEqual(str(id2type[45]), "NoneType")	# If
        self.assertEqual(str(id2type[46]), "bool")	# Compare  (line 17)
        self.assertEqual(str(id2type[47]), "NoneType")	# Attribute self.h_length (line 17)
        self.assertEqual(str(id2type[48]), "class AttLoc")	# Name self (line 17)
        self.assertEqual(str(id2type[52]), "NoneType")	# Constant None (line 17)
        self.assertEqual(str(id2type[53]), "NoneType")	# Assign
        self.assertEqual(str(id2type[54]), "int")	# Attribute self.h_length (line 18)
        self.assertEqual(str(id2type[55]), "class AttLoc")	# Name self (line 18)
        self.assertEqual(str(id2type[58]), "int")	# Subscript self.enc_h.shape[1] (line 18)
        self.assertEqual(str(id2type[59]), "(int, int, int)")	# Attribute self.enc_h.shape (line 18)
        self.assertEqual(str(id2type[60]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 18)
        self.assertEqual(str(id2type[61]), "class AttLoc")	# Name self (line 18)
        self.assertEqual(str(id2type[66]), "int")	# Constant 1 (line 18)
        self.assertEqual(str(id2type[68]), "NoneType")	# If
        self.assertEqual(str(id2type[69]), "bool")	# Compare  (line 21)
        self.assertEqual(str(id2type[70]), "NoneType")	# Attribute self.pre_compute_enc_h (line 21)
        self.assertEqual(str(id2type[71]), "class AttLoc")	# Name self (line 21)
        self.assertEqual(str(id2type[75]), "NoneType")	# Constant None (line 21)
        self.assertEqual(str(id2type[76]), "NoneType")	# Assign
        self.assertEqual(str(id2type[77]), "Variable(float32, (3, 4, 5))")	# Attribute self.pre_compute_enc_h (line 23)
        self.assertEqual(str(id2type[78]), "class AttLoc")	# Name self (line 23)
        self.assertEqual(str(id2type[81]), "Variable(float32, (3, 4, 5))")	# Call linear_tensor_3d(self.mlp_enc, self.enc_h) (line 23)
        self.assertEqual(str(id2type[82]), "class Linear -> Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 5))")	# Name linear_tensor_3d (line 23)
        self.assertEqual(str(id2type[84]), "class Linear")	# Attribute self.mlp_enc (line 23)
        self.assertEqual(str(id2type[85]), "class AttLoc")	# Name self (line 23)
        self.assertEqual(str(id2type[88]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 23)
        self.assertEqual(str(id2type[89]), "class AttLoc")	# Name self (line 23)
        self.assertEqual(str(id2type[92]), "NoneType")	# If
        self.assertEqual(str(id2type[93]), "bool")	# Compare  (line 25)
        self.assertEqual(str(id2type[94]), "NoneType")	# Name dec_z (line 25)
        self.assertEqual(str(id2type[97]), "NoneType")	# Constant None (line 25)
        self.assertEqual(str(id2type[98]), "NoneType")	# Assign
        self.assertEqual(str(id2type[99]), "Variable(float32, (3, 4))")	# Name dec_z_new (line 26)
        self.assertEqual(str(id2type[101]), "Variable(float32, (3, 4))")	# Call chainer.Variable(self.xp.zeros((batch, self.dunits), dtype=np.float32)) (line 26)
        self.assertEqual(str(id2type[102]), "ndarray(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute chainer.Variable (line 26)
        self.assertEqual(str(id2type[106]), "ndarray(float32, (3, 4))")	# Call self.xp.zeros((batch, self.dunits), dtype=np.float32) (line 26)
        self.assertEqual(str(id2type[107]), "(int, int) -> ndarray(float32, (3, 4))")	# Attribute self.xp.zeros (line 26)
        self.assertEqual(str(id2type[108]), "class module")	# Attribute self.xp (line 26)
        self.assertEqual(str(id2type[109]), "class AttLoc")	# Name self (line 26)
        self.assertEqual(str(id2type[113]), "(int, int)")	# Tuple (batch, self.dunits) (line 27)
        self.assertEqual(str(id2type[114]), "int")	# Name batch (line 27)
        self.assertEqual(str(id2type[116]), "int")	# Attribute self.dunits (line 27)
        self.assertEqual(str(id2type[117]), "class AttLoc")	# Name self (line 27)
        self.assertEqual(str(id2type[122]), "dtype(float32)")	# Attribute np.float32 (line 27)
        self.assertEqual(str(id2type[126]), "NoneType")	# Assign
        self.assertEqual(str(id2type[127]), "a9")	# Name dec_z_new (line 29)
        self.assertEqual(str(id2type[129]), "a7 (from line 29)")	# Call F.reshape(dec_z, (batch, self.dunits)) (line 29)
        self.assertEqual(str(id2type[130]), "NoneType -> (int, int) -> a7 (from line 29)")	# Attribute F.reshape (line 29)
        self.assertEqual(str(id2type[134]), "NoneType")	# Name dec_z (line 29)
        self.assertEqual(str(id2type[136]), "(int, int)")	# Tuple (batch, self.dunits) (line 29)
        self.assertEqual(str(id2type[137]), "int")	# Name batch (line 29)
        self.assertEqual(str(id2type[139]), "int")	# Attribute self.dunits (line 29)
        self.assertEqual(str(id2type[140]), "class AttLoc")	# Name self (line 29)
        self.assertEqual(str(id2type[144]), "NoneType")	# If
        self.assertEqual(str(id2type[145]), "bool")	# Compare  (line 32)
        self.assertEqual(str(id2type[146]), "NoneType")	# Name att_prev (line 32)
        self.assertEqual(str(id2type[149]), "NoneType")	# Constant None (line 32)
        self.assertEqual(str(id2type[150]), "NoneType")	# Assign
        self.assertEqual(str(id2type[151]), "ndarray(float32, (None,)) list")	# Name att_prev (line 33)
        self.assertEqual(str(id2type[153]), "ndarray(float32, (None,)) list")	# ListComp  (line 33)
        self.assertEqual(str(id2type[154]), "ndarray(float32, (None,))")	# Call self.xp.full(hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) (line 33)
        self.assertEqual(str(id2type[155]), "int -> float -> ndarray(float32, (None,))")	# Attribute self.xp.full (line 33)
        self.assertEqual(str(id2type[156]), "class module")	# Attribute self.xp (line 33)
        self.assertEqual(str(id2type[157]), "class AttLoc")	# Name self (line 33)
        self.assertEqual(str(id2type[161]), "int")	# Subscript hh.shape[0] (line 34)
        self.assertEqual(str(id2type[162]), "(int, int)")	# Attribute hh.shape (line 34)
        self.assertEqual(str(id2type[163]), "ndarray(float32, (None, 3))")	# Name hh (line 34)
        self.assertEqual(str(id2type[167]), "int")	# Constant 0 (line 34)
        self.assertEqual(str(id2type[169]), "float")	# BinOp 1.0 / hh.shape[0] (line 34)
        self.assertEqual(str(id2type[170]), "float")	# Constant 1.0 (line 34)
        self.assertEqual(str(id2type[171]), "float -> int -> float")	# Div
        self.assertEqual(str(id2type[172]), "int")	# Subscript hh.shape[0] (line 34)
        self.assertEqual(str(id2type[173]), "(int, int)")	# Attribute hh.shape (line 34)
        self.assertEqual(str(id2type[174]), "ndarray(float32, (None, 3))")	# Name hh (line 34)
        self.assertEqual(str(id2type[178]), "int")	# Constant 0 (line 34)
        self.assertEqual(str(id2type[181]), "dtype(float32)")	# Attribute np.float32 (line 34)
        self.assertEqual(str(id2type[186]), "ndarray(float32, (None, 3))")	# Name hh (line 34)
        self.assertEqual(str(id2type[188]), "ndarray(float32, (None, 3)) list")	# Name enc_hs (line 34)
        self.assertEqual(str(id2type[190]), "NoneType")	# Assign
        self.assertEqual(str(id2type[191]), "Variable(float32, (None,)) list")	# Name att_prev (line 35)
        self.assertEqual(str(id2type[193]), "Variable(float32, (None,)) list")	# ListComp  (line 35)
        self.assertEqual(str(id2type[194]), "Variable(float32, (None,))")	# Call chainer.Variable(att) (line 35)
        self.assertEqual(str(id2type[195]), "ndarray(float32, (None,)) -> Variable(float32, (None,))")	# Attribute chainer.Variable (line 35)
        self.assertEqual(str(id2type[199]), "ndarray(float32, (None,))")	# Name att (line 35)
        self.assertEqual(str(id2type[202]), "ndarray(float32, (None,))")	# Name att (line 35)
        self.assertEqual(str(id2type[204]), "ndarray(float32, (None,)) list")	# Name att_prev (line 35)
        self.assertEqual(str(id2type[206]), "NoneType")	# Assign
        self.assertEqual(str(id2type[207]), "Variable(float32, (None, None))")	# Name att_prev (line 36)
        self.assertEqual(str(id2type[209]), "Variable(float32, (None, None))")	# Call F.pad_sequence(att_prev) (line 36)
        self.assertEqual(str(id2type[210]), "Variable(float32, (None,)) list -> Variable(float32, (None, None))")	# Attribute F.pad_sequence (line 36)
        self.assertEqual(str(id2type[214]), "Variable(float32, (None,)) list")	# Name att_prev (line 36)
        self.assertEqual(str(id2type[216]), "NoneType")	# Assign
        self.assertEqual(str(id2type[217]), "Variable(float32, (3, 7, 1, 4))")	# Name att_conv (line 40)
        self.assertEqual(str(id2type[219]), "Variable(float32, (3, 7, 1, 4))")	# Call self.loc_conv(F.reshape(att_prev, (batch, 1, 1, self.h_length))) (line 40)
        self.assertEqual(str(id2type[220]), "Variable(float32, (3, 1, 1, 4)) -> Variable(float32, (3, 7, 1, 4))")	# Attribute self.loc_conv (line 40)
        self.assertEqual(str(id2type[221]), "class AttLoc")	# Name self (line 40)
        self.assertEqual(str(id2type[224]), "Variable(float32, (3, 1, 1, 4))")	# Call F.reshape(att_prev, (batch, 1, 1, self.h_length)) (line 41)
        self.assertEqual(str(id2type[225]), "Variable(float32, (None, None)) -> (int, int, int, int) -> Variable(float32, (3, 1, 1, 4))")	# Attribute F.reshape (line 41)
        self.assertEqual(str(id2type[229]), "Variable(float32, (None, None))")	# Name att_prev (line 41)
        self.assertEqual(str(id2type[231]), "(int, int, int, int)")	# Tuple (batch, 1, 1, self.h_length) (line 41)
        self.assertEqual(str(id2type[232]), "int")	# Name batch (line 41)
        self.assertEqual(str(id2type[234]), "int")	# Constant 1 (line 41)
        self.assertEqual(str(id2type[235]), "int")	# Constant 1 (line 41)
        self.assertEqual(str(id2type[236]), "int")	# Attribute self.h_length (line 41)
        self.assertEqual(str(id2type[237]), "class AttLoc")	# Name self (line 41)
        self.assertEqual(str(id2type[241]), "NoneType")	# Assign
        self.assertEqual(str(id2type[242]), "Variable(float32, (3, 4, 7))")	# Name att_conv (line 43)
        self.assertEqual(str(id2type[244]), "Variable(float32, (3, 4, 7))")	# Call F.swapaxes(F.squeeze(att_conv, axis=2), 1, 2) (line 43)
        self.assertEqual(str(id2type[245]), "Variable(float32, (3, 7, 4)) -> int -> int -> Variable(float32, (3, 4, 7))")	# Attribute F.swapaxes (line 43)
        self.assertEqual(str(id2type[249]), "Variable(float32, (3, 7, 4))")	# Call F.squeeze(att_conv, axis=2) (line 43)
        self.assertEqual(str(id2type[250]), "Variable(float32, (3, 7, 1, 4)) -> Variable(float32, (3, 7, 4))")	# Attribute F.squeeze (line 43)
        self.assertEqual(str(id2type[254]), "Variable(float32, (3, 7, 1, 4))")	# Name att_conv (line 43)
        self.assertEqual(str(id2type[257]), "int")	# Constant 2 (line 43)
        self.assertEqual(str(id2type[258]), "int")	# Constant 1 (line 43)
        self.assertEqual(str(id2type[259]), "int")	# Constant 2 (line 43)
        self.assertEqual(str(id2type[260]), "NoneType")	# Assign
        self.assertEqual(str(id2type[261]), "Variable(float32, (3, 4, 5))")	# Name att_conv (line 45)
        self.assertEqual(str(id2type[263]), "Variable(float32, (3, 4, 5))")	# Call linear_tensor_3d(self.mlp_att, att_conv) (line 45)
        self.assertEqual(str(id2type[264]), "class Linear -> Variable(float32, (3, 4, 7)) -> Variable(float32, (3, 4, 5))")	# Name linear_tensor_3d (line 45)
        self.assertEqual(str(id2type[266]), "class Linear")	# Attribute self.mlp_att (line 45)
        self.assertEqual(str(id2type[267]), "class AttLoc")	# Name self (line 45)
        self.assertEqual(str(id2type[270]), "Variable(float32, (3, 4, 7))")	# Name att_conv (line 45)
        self.assertEqual(str(id2type[272]), "NoneType")	# Assign
        self.assertEqual(str(id2type[273]), "Variable(float32, (3, 4, 5))")	# Name dec_z_tiled (line 48)
        self.assertEqual(str(id2type[275]), "Variable(float32, (3, 4, 5))")	# Call F.broadcast_to(F.expand_dims(self.mlp_dec(dec_z_new), 1), self.pre_compute_enc_h.shape) (line 48)
        self.assertEqual(str(id2type[276]), "Variable(float32, (3, 1, 5)) -> (int, int, int) -> Variable(float32, (3, 4, 5))")	# Attribute F.broadcast_to (line 48)
        self.assertEqual(str(id2type[280]), "Variable(float32, (3, 1, 5))")	# Call F.expand_dims(self.mlp_dec(dec_z_new), 1) (line 49)
        self.assertEqual(str(id2type[281]), "Variable(float32, (3, 5)) -> int -> Variable(float32, (3, 1, 5))")	# Attribute F.expand_dims (line 49)
        self.assertEqual(str(id2type[285]), "Variable(float32, (3, 5))")	# Call self.mlp_dec(dec_z_new) (line 49)
        self.assertEqual(str(id2type[286]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 5))")	# Attribute self.mlp_dec (line 49)
        self.assertEqual(str(id2type[287]), "class AttLoc")	# Name self (line 49)
        self.assertEqual(str(id2type[290]), "Variable(float32, (3, 4))")	# Name dec_z_new (line 49)
        self.assertEqual(str(id2type[292]), "int")	# Constant 1 (line 49)
        self.assertEqual(str(id2type[293]), "(int, int, int)")	# Attribute self.pre_compute_enc_h.shape (line 49)
        self.assertEqual(str(id2type[294]), "Variable(float32, (3, 4, 5))")	# Attribute self.pre_compute_enc_h (line 49)
        self.assertEqual(str(id2type[295]), "class AttLoc")	# Name self (line 49)
        self.assertEqual(str(id2type[299]), "NoneType")	# Assign
        self.assertEqual(str(id2type[300]), "Variable(float32, (3, 4))")	# Name e (line 54)
        self.assertEqual(str(id2type[302]), "Variable(float32, (3, 4))")	# Call F.squeeze(linear_tensor_3d(self.gvec, F.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)), axis=2) (line 54)
        self.assertEqual(str(id2type[303]), "Variable(float32, (3, 4, 1)) -> Variable(float32, (3, 4))")	# Attribute F.squeeze (line 54)
        self.assertEqual(str(id2type[307]), "Variable(float32, (3, 4, 1))")	# Call linear_tensor_3d(self.gvec, F.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)) (line 54)
        self.assertEqual(str(id2type[308]), "class Linear -> Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 1))")	# Name linear_tensor_3d (line 54)
        self.assertEqual(str(id2type[310]), "class Linear")	# Attribute self.gvec (line 54)
        self.assertEqual(str(id2type[311]), "class AttLoc")	# Name self (line 54)
        self.assertEqual(str(id2type[314]), "Variable(float32, (3, 4, 5))")	# Call F.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled) (line 54)
        self.assertEqual(str(id2type[315]), "Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5))")	# Attribute F.tanh (line 54)
        self.assertEqual(str(id2type[319]), "Variable(float32, (3, 4, 5))")	# BinOp att_conv + self.pre_compute_enc_h + dec_z_tiled (line 55)
        self.assertEqual(str(id2type[320]), "Variable(float32, (3, 4, 5))")	# BinOp att_conv + self.pre_compute_enc_h (line 55)
        self.assertEqual(str(id2type[321]), "Variable(float32, (3, 4, 5))")	# Name att_conv (line 55)
        self.assertEqual(str(id2type[323]), "Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5))")	# Add
        self.assertEqual(str(id2type[324]), "Variable(float32, (3, 4, 5))")	# Attribute self.pre_compute_enc_h (line 55)
        self.assertEqual(str(id2type[325]), "class AttLoc")	# Name self (line 55)
        self.assertEqual(str(id2type[328]), "Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 5))")	# Add
        self.assertEqual(str(id2type[329]), "Variable(float32, (3, 4, 5))")	# Name dec_z_tiled (line 55)
        self.assertEqual(str(id2type[332]), "int")	# Constant 2 (line 55)
        self.assertEqual(str(id2type[333]), "NoneType")	# Assign
        self.assertEqual(str(id2type[334]), "Variable(float32, (3, 4))")	# Name w (line 59)
        self.assertEqual(str(id2type[336]), "Variable(float32, (3, 4))")	# Call F.softmax(scaling * e) (line 59)
        self.assertEqual(str(id2type[337]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute F.softmax (line 59)
        self.assertEqual(str(id2type[341]), "Variable(float32, (3, 4))")	# BinOp scaling * e (line 59)
        self.assertEqual(str(id2type[342]), "float")	# Name scaling (line 59)
        self.assertEqual(str(id2type[344]), "float -> Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Mult
        self.assertEqual(str(id2type[345]), "Variable(float32, (3, 4))")	# Name e (line 59)
        self.assertEqual(str(id2type[347]), "NoneType")	# Assign
        self.assertEqual(str(id2type[348]), "Variable(float32, (3, 3))")	# Name c (line 63)
        self.assertEqual(str(id2type[350]), "Variable(float32, (3, 3))")	# Call F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1) (line 63)
        self.assertEqual(str(id2type[351]), "Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 3))")	# Attribute F.sum (line 63)
        self.assertEqual(str(id2type[355]), "Variable(float32, (3, 4, 3))")	# BinOp self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape) (line 63)
        self.assertEqual(str(id2type[356]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 63)
        self.assertEqual(str(id2type[357]), "class AttLoc")	# Name self (line 63)
        self.assertEqual(str(id2type[360]), "Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 3))")	# Mult
        self.assertEqual(str(id2type[361]), "Variable(float32, (3, 4, 3))")	# Call F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape) (line 63)
        self.assertEqual(str(id2type[362]), "Variable(float32, (3, 4, 1)) -> (int, int, int) -> Variable(float32, (3, 4, 3))")	# Attribute F.broadcast_to (line 63)
        self.assertEqual(str(id2type[366]), "Variable(float32, (3, 4, 1))")	# Call F.expand_dims(w, 2) (line 63)
        self.assertEqual(str(id2type[367]), "Variable(float32, (3, 4)) -> int -> Variable(float32, (3, 4, 1))")	# Attribute F.expand_dims (line 63)
        self.assertEqual(str(id2type[371]), "Variable(float32, (3, 4))")	# Name w (line 63)
        self.assertEqual(str(id2type[373]), "int")	# Constant 2 (line 63)
        self.assertEqual(str(id2type[374]), "(int, int, int)")	# Attribute self.enc_h.shape (line 63)
        self.assertEqual(str(id2type[375]), "Variable(float32, (3, 4, 3))")	# Attribute self.enc_h (line 63)
        self.assertEqual(str(id2type[376]), "class AttLoc")	# Name self (line 63)
        self.assertEqual(str(id2type[381]), "int")	# Constant 1 (line 63)
        self.assertEqual(str(id2type[382]), "(Variable(float32, (3, 3)), Variable(float32, (3, 4)))")	# Return
        self.assertEqual(str(id2type[383]), "(Variable(float32, (3, 3)), Variable(float32, (3, 4)))")	# Tuple (c, w) (line 65)
        self.assertEqual(str(id2type[384]), "Variable(float32, (3, 3))")	# Name c (line 65)
        self.assertEqual(str(id2type[386]), "Variable(float32, (3, 4))")	# Name w (line 65)
        self.assertEqual(str(id2type[389]), "class Linear -> Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 5))")	# FunctionDef linear_tensor_3d (line 1)
        self.assertEqual(str(id2type[395]), "NoneType")	# Expr
        self.assertEqual(str(id2type[396]), "string")	# Constant "..." (line 8)
        self.assertEqual(str(id2type[397]), "Variable(float32, (3, 4, 5))")	# Return
        self.assertEqual(str(id2type[398]), "Variable(float32, (3, 4, 5))")	# Call linear(x, n_batch_axes=2) (line 9)
        self.assertEqual(str(id2type[399]), "Variable(float32, (3, 4, 3)) -> Variable(float32, (3, 4, 5))")	# Name linear (line 9)
        self.assertEqual(str(id2type[401]), "Variable(float32, (3, 4, 3))")	# Name x (line 9)
        self.assertEqual(str(id2type[404]), "int")	# Constant 2 (line 9)
        self.assertEqual(str(id2type[405]), "class Linear -> Variable(float32, (3, 4, 7)) -> Variable(float32, (3, 4, 5))")	# FunctionDef linear_tensor_3d (line 1)
        self.assertEqual(str(id2type[411]), "NoneType")	# Expr
        self.assertEqual(str(id2type[412]), "string")	# Constant "..." (line 8)
        self.assertEqual(str(id2type[413]), "Variable(float32, (3, 4, 5))")	# Return
        self.assertEqual(str(id2type[414]), "Variable(float32, (3, 4, 5))")	# Call linear(x, n_batch_axes=2) (line 9)
        self.assertEqual(str(id2type[415]), "Variable(float32, (3, 4, 7)) -> Variable(float32, (3, 4, 5))")	# Name linear (line 9)
        self.assertEqual(str(id2type[417]), "Variable(float32, (3, 4, 7))")	# Name x (line 9)
        self.assertEqual(str(id2type[420]), "int")	# Constant 2 (line 9)
        self.assertEqual(str(id2type[421]), "class Linear -> Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 1))")	# FunctionDef linear_tensor_3d (line 1)
        self.assertEqual(str(id2type[427]), "NoneType")	# Expr
        self.assertEqual(str(id2type[428]), "string")	# Constant "..." (line 8)
        self.assertEqual(str(id2type[429]), "Variable(float32, (3, 4, 1))")	# Return
        self.assertEqual(str(id2type[430]), "Variable(float32, (3, 4, 1))")	# Call linear(x, n_batch_axes=2) (line 9)
        self.assertEqual(str(id2type[431]), "Variable(float32, (3, 4, 5)) -> Variable(float32, (3, 4, 1))")	# Name linear (line 9)
        self.assertEqual(str(id2type[433]), "Variable(float32, (3, 4, 5))")	# Name x (line 9)
        self.assertEqual(str(id2type[436]), "int")	# Constant 2 (line 9)
        # === END ASSERTIONS for AttLoc ===


    def test_StatelessLSTM(self):
        model, forward_args = gen_StatelessLSTM_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for StatelessLSTM ===
        self.assertEqual(str(id2type[1]), "class StatelessLSTM -> Variable(float32, (3, 4)) -> Variable(float32, (3, 4)) -> Variable(float32, (3, 7)) -> (Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[11]), "NoneType")	# Expr
        self.assertEqual(str(id2type[12]), "string")	# Constant "..." (line 14)
        self.assertEqual(str(id2type[13]), "NoneType")	# Assign
        self.assertEqual(str(id2type[14]), "Variable(float32, (3, 16))")	# Name lstm_in (line 22)
        self.assertEqual(str(id2type[16]), "Variable(float32, (3, 16))")	# Call self.upward(x) (line 22)
        self.assertEqual(str(id2type[17]), "Variable(float32, (3, 7)) -> Variable(float32, (3, 16))")	# Attribute self.upward (line 22)
        self.assertEqual(str(id2type[18]), "class StatelessLSTM")	# Name self (line 22)
        self.assertEqual(str(id2type[21]), "Variable(float32, (3, 7))")	# Name x (line 22)
        self.assertEqual(str(id2type[23]), "NoneType")	# If
        self.assertEqual(str(id2type[24]), "bool")	# Compare  (line 23)
        self.assertEqual(str(id2type[25]), "Variable(float32, (3, 4))")	# Name h (line 23)
        self.assertEqual(str(id2type[28]), "NoneType")	# Constant None (line 23)
        self.assertEqual(str(id2type[29]), "NoneType")	# AugAssign
        self.assertEqual(str(id2type[30]), "Variable(float32, (3, 16))")	# Name lstm_in (line 24)
        self.assertEqual(str(id2type[32]), "Variable(float32, (3, 16)) -> Variable(float32, (3, 16)) -> Variable(float32, (3, 16))")	# Add
        self.assertEqual(str(id2type[33]), "Variable(float32, (3, 16))")	# Call self.lateral(h) (line 24)
        self.assertEqual(str(id2type[34]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 16))")	# Attribute self.lateral (line 24)
        self.assertEqual(str(id2type[35]), "class StatelessLSTM")	# Name self (line 24)
        self.assertEqual(str(id2type[38]), "Variable(float32, (3, 4))")	# Name h (line 24)
        self.assertEqual(str(id2type[40]), "NoneType")	# If
        self.assertEqual(str(id2type[41]), "bool")	# Compare  (line 25)
        self.assertEqual(str(id2type[42]), "Variable(float32, (3, 4))")	# Name c (line 25)
        self.assertEqual(str(id2type[45]), "NoneType")	# Constant None (line 25)
        self.assertEqual(str(id2type[46]), "NoneType")	# Assign
        self.assertEqual(str(id2type[47]), "Variable(float32, (3, 4))")	# Name c (line 31)
        self.assertEqual(str(id2type[49]), "Variable(float32, (3, 4))")	# Call variable.Variable(self.xp.zeros((x.shape[0], self.state_size), dtype=self.xp.float32)) (line 31)
        self.assertEqual(str(id2type[50]), "ndarray(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute variable.Variable (line 31)
        self.assertEqual(str(id2type[54]), "ndarray(float32, (3, 4))")	# Call self.xp.zeros((x.shape[0], self.state_size), dtype=self.xp.float32) (line 32)
        self.assertEqual(str(id2type[55]), "(int, int) -> ndarray(float32, (3, 4))")	# Attribute self.xp.zeros (line 32)
        self.assertEqual(str(id2type[56]), "class module")	# Attribute self.xp (line 32)
        self.assertEqual(str(id2type[57]), "class StatelessLSTM")	# Name self (line 32)
        self.assertEqual(str(id2type[61]), "(int, int)")	# Tuple (x.shape[0], self.state_size) (line 32)
        self.assertEqual(str(id2type[62]), "int")	# Subscript x.shape[0] (line 32)
        self.assertEqual(str(id2type[63]), "(int, int)")	# Attribute x.shape (line 32)
        self.assertEqual(str(id2type[64]), "Variable(float32, (3, 7))")	# Name x (line 32)
        self.assertEqual(str(id2type[68]), "int")	# Constant 0 (line 32)
        self.assertEqual(str(id2type[70]), "int")	# Attribute self.state_size (line 32)
        self.assertEqual(str(id2type[71]), "class StatelessLSTM")	# Name self (line 32)
        self.assertEqual(str(id2type[76]), "dtype(float32)")	# Attribute self.xp.float32 (line 32)
        self.assertEqual(str(id2type[77]), "class module")	# Attribute self.xp (line 32)
        self.assertEqual(str(id2type[78]), "class StatelessLSTM")	# Name self (line 32)
        self.assertEqual(str(id2type[82]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Return
        self.assertEqual(str(id2type[83]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Call lstm_forward(c, lstm_in) (line 34)
        self.assertEqual(str(id2type[84]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 16)) -> (Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Name lstm_forward (line 34)
        self.assertEqual(str(id2type[86]), "Variable(float32, (3, 4))")	# Name c (line 34)
        self.assertEqual(str(id2type[88]), "Variable(float32, (3, 16))")	# Name lstm_in (line 34)
        self.assertEqual(str(id2type[90]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 16)) -> (Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# FunctionDef lstm_forward (line 1)
        self.assertEqual(str(id2type[96]), "NoneType")	# Assign
        self.assertEqual(str(id2type[97]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Tuple (a, i, f, o) (line 2)
        self.assertEqual(str(id2type[98]), "Variable(float32, (3, 4))")	# Name a (line 2)
        self.assertEqual(str(id2type[100]), "Variable(float32, (3, 4))")	# Name i (line 2)
        self.assertEqual(str(id2type[102]), "Variable(float32, (3, 4))")	# Name f (line 2)
        self.assertEqual(str(id2type[104]), "Variable(float32, (3, 4))")	# Name o (line 2)
        self.assertEqual(str(id2type[107]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Call _extract_gates(x) (line 2)
        self.assertEqual(str(id2type[108]), "Variable(float32, (3, 16)) -> (Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Name _extract_gates (line 2)
        self.assertEqual(str(id2type[110]), "Variable(float32, (3, 16))")	# Name x (line 2)
        self.assertEqual(str(id2type[112]), "NoneType")	# Assign
        self.assertEqual(str(id2type[113]), "int")	# Name batch (line 3)
        self.assertEqual(str(id2type[115]), "int")	# Call len(x) (line 3)
        self.assertEqual(str(id2type[116]), "Variable(float32, (3, 16)) -> int")	# Name len (line 3)
        self.assertEqual(str(id2type[118]), "Variable(float32, (3, 16))")	# Name x (line 3)
        self.assertEqual(str(id2type[120]), "NoneType")	# Assign
        self.assertEqual(str(id2type[121]), "Variable(float32, (3, 4))")	# Name a (line 5)
        self.assertEqual(str(id2type[123]), "Variable(float32, (3, 4))")	# Call F.tanh(a) (line 5)
        self.assertEqual(str(id2type[124]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute F.tanh (line 5)
        self.assertEqual(str(id2type[128]), "Variable(float32, (3, 4))")	# Name a (line 5)
        self.assertEqual(str(id2type[130]), "NoneType")	# Assign
        self.assertEqual(str(id2type[131]), "Variable(float32, (3, 4))")	# Name i (line 6)
        self.assertEqual(str(id2type[133]), "Variable(float32, (3, 4))")	# Call F.sigmoid(i) (line 6)
        self.assertEqual(str(id2type[134]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute F.sigmoid (line 6)
        self.assertEqual(str(id2type[138]), "Variable(float32, (3, 4))")	# Name i (line 6)
        self.assertEqual(str(id2type[140]), "NoneType")	# Assign
        self.assertEqual(str(id2type[141]), "Variable(float32, (3, 4))")	# Name f (line 7)
        self.assertEqual(str(id2type[143]), "Variable(float32, (3, 4))")	# Call F.sigmoid(f) (line 7)
        self.assertEqual(str(id2type[144]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute F.sigmoid (line 7)
        self.assertEqual(str(id2type[148]), "Variable(float32, (3, 4))")	# Name f (line 7)
        self.assertEqual(str(id2type[150]), "NoneType")	# Assign
        self.assertEqual(str(id2type[151]), "Variable(float32, (3, 4))")	# Name o (line 8)
        self.assertEqual(str(id2type[153]), "Variable(float32, (3, 4))")	# Call F.sigmoid(o) (line 8)
        self.assertEqual(str(id2type[154]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute F.sigmoid (line 8)
        self.assertEqual(str(id2type[158]), "Variable(float32, (3, 4))")	# Name o (line 8)
        self.assertEqual(str(id2type[160]), "NoneType")	# Assign
        self.assertEqual(str(id2type[161]), "Variable(float32, (3, 4))")	# Name c_next (line 10)
        self.assertEqual(str(id2type[163]), "Variable(float32, (3, 4))")	# BinOp a * i + f * c_prev (line 10)
        self.assertEqual(str(id2type[164]), "Variable(float32, (3, 4))")	# BinOp a * i (line 10)
        self.assertEqual(str(id2type[165]), "Variable(float32, (3, 4))")	# Name a (line 10)
        self.assertEqual(str(id2type[167]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Mult
        self.assertEqual(str(id2type[168]), "Variable(float32, (3, 4))")	# Name i (line 10)
        self.assertEqual(str(id2type[170]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Add
        self.assertEqual(str(id2type[171]), "Variable(float32, (3, 4))")	# BinOp f * c_prev (line 10)
        self.assertEqual(str(id2type[172]), "Variable(float32, (3, 4))")	# Name f (line 10)
        self.assertEqual(str(id2type[174]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Mult
        self.assertEqual(str(id2type[175]), "Variable(float32, (3, 4))")	# Name c_prev (line 10)
        self.assertEqual(str(id2type[177]), "NoneType")	# Assign
        self.assertEqual(str(id2type[178]), "Variable(float32, (3, 4))")	# Name h (line 11)
        self.assertEqual(str(id2type[180]), "Variable(float32, (3, 4))")	# BinOp o * F.tanh(c_next) (line 11)
        self.assertEqual(str(id2type[181]), "Variable(float32, (3, 4))")	# Name o (line 11)
        self.assertEqual(str(id2type[183]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Mult
        self.assertEqual(str(id2type[184]), "Variable(float32, (3, 4))")	# Call F.tanh(c_next) (line 11)
        self.assertEqual(str(id2type[185]), "Variable(float32, (3, 4)) -> Variable(float32, (3, 4))")	# Attribute F.tanh (line 11)
        self.assertEqual(str(id2type[189]), "Variable(float32, (3, 4))")	# Name c_next (line 11)
        self.assertEqual(str(id2type[191]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Return
        self.assertEqual(str(id2type[192]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Tuple (c_next, h) (line 12)
        self.assertEqual(str(id2type[193]), "Variable(float32, (3, 4))")	# Name c_next (line 12)
        self.assertEqual(str(id2type[195]), "Variable(float32, (3, 4))")	# Name h (line 12)
        self.assertEqual(str(id2type[198]), "Variable(float32, (3, 16)) -> (Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# FunctionDef _extract_gates (line 1)
        self.assertEqual(str(id2type[202]), "NoneType")	# Assign
        self.assertEqual(str(id2type[203]), "Variable(float32, (3, 4, 4))")	# Name r (line 2)
        self.assertEqual(str(id2type[205]), "Variable(float32, (3, 4, 4))")	# Call F.reshape(x, (len(x), x.shape[1] // 4, 4) + x.shape[2::]) (line 2)
        self.assertEqual(str(id2type[206]), "Variable(float32, (3, 16)) -> (int, int, int) -> Variable(float32, (3, 4, 4))")	# Attribute F.reshape (line 2)
        self.assertEqual(str(id2type[210]), "Variable(float32, (3, 16))")	# Name x (line 2)
        self.assertEqual(str(id2type[212]), "(int, int, int)")	# BinOp (len(x), x.shape[1] // 4, 4) + x.shape[2::] (line 2)
        self.assertEqual(str(id2type[213]), "(int, int, int)")	# Tuple (len(x), x.shape[1] // 4, 4) (line 2)
        self.assertEqual(str(id2type[214]), "int")	# Call len(x) (line 2)
        self.assertEqual(str(id2type[215]), "Variable(float32, (3, 16)) -> int")	# Name len (line 2)
        self.assertEqual(str(id2type[217]), "Variable(float32, (3, 16))")	# Name x (line 2)
        self.assertEqual(str(id2type[219]), "int")	# BinOp x.shape[1] // 4 (line 2)
        self.assertEqual(str(id2type[220]), "int")	# Subscript x.shape[1] (line 2)
        self.assertEqual(str(id2type[221]), "(int, int)")	# Attribute x.shape (line 2)
        self.assertEqual(str(id2type[222]), "Variable(float32, (3, 16))")	# Name x (line 2)
        self.assertEqual(str(id2type[226]), "int")	# Constant 1 (line 2)
        self.assertEqual(str(id2type[228]), "int -> int -> int")	# FloorDiv
        self.assertEqual(str(id2type[229]), "int")	# Constant 4 (line 2)
        self.assertEqual(str(id2type[230]), "int")	# Constant 4 (line 2)
        self.assertEqual(str(id2type[232]), "(int, int, int) -> () -> (int, int, int)")	# Add
        self.assertEqual(str(id2type[233]), "()")	# Subscript x.shape[2::] (line 2)
        self.assertEqual(str(id2type[234]), "(int, int)")	# Attribute x.shape (line 2)
        self.assertEqual(str(id2type[235]), "Variable(float32, (3, 16))")	# Name x (line 2)
        self.assertEqual(str(id2type[239]), "int")	# Constant 2 (line 2)
        self.assertEqual(str(id2type[241]), "NoneType")	# Assign
        self.assertEqual(str(id2type[242]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Name r (line 3)
        self.assertEqual(str(id2type[244]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Call F.separate(r, axis=2) (line 3)
        self.assertEqual(str(id2type[245]), "Variable(float32, (3, 4, 4)) -> (Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Attribute F.separate (line 3)
        self.assertEqual(str(id2type[249]), "Variable(float32, (3, 4, 4))")	# Name r (line 3)
        self.assertEqual(str(id2type[252]), "int")	# Constant 2 (line 3)
        self.assertEqual(str(id2type[253]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Return
        self.assertEqual(str(id2type[254]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Tuple (r[0], r[1], r[2], r[3]) (line 4)
        self.assertEqual(str(id2type[255]), "Variable(float32, (3, 4))")	# Subscript r[0] (line 4)
        self.assertEqual(str(id2type[256]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Name r (line 4)
        self.assertEqual(str(id2type[259]), "int")	# Constant 0 (line 4)
        self.assertEqual(str(id2type[261]), "Variable(float32, (3, 4))")	# Subscript r[1] (line 4)
        self.assertEqual(str(id2type[262]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Name r (line 4)
        self.assertEqual(str(id2type[265]), "int")	# Constant 1 (line 4)
        self.assertEqual(str(id2type[267]), "Variable(float32, (3, 4))")	# Subscript r[2] (line 4)
        self.assertEqual(str(id2type[268]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Name r (line 4)
        self.assertEqual(str(id2type[271]), "int")	# Constant 2 (line 4)
        self.assertEqual(str(id2type[273]), "Variable(float32, (3, 4))")	# Subscript r[3] (line 4)
        self.assertEqual(str(id2type[274]), "(Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)), Variable(float32, (3, 4)))")	# Name r (line 4)
        self.assertEqual(str(id2type[277]), "int")	# Constant 3 (line 4)
        # === END ASSERTIONS for StatelessLSTM ===


    def test_VGG2L(self):
        model, forward_args = gen_VGG2L_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for VGG2L ===
        self.assertEqual(str(id2type[1]), "class VGG2L -> [ndarray(float32, (4, 5)), ndarray(float32, (2, 5)), ndarray(float32, (2, 5))] -> ndarray(int64, (3,)) -> (Variable(float32, (None, 256)) list, ndarray(int64, (3,)))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[9]), "NoneType")	# Expr
        self.assertEqual(str(id2type[10]), "string")	# Constant "..." (line 7)
        self.assertEqual(str(id2type[11]), "NoneType")	# Expr
        self.assertEqual(str(id2type[12]), "NoneType")	# Call logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens)) (line 8)
        self.assertEqual(str(id2type[33]), "NoneType")	# Assign
        self.assertEqual(str(id2type[34]), "Variable(float32, (3, 4, 5))")	# Name xs (line 11)
        self.assertEqual(str(id2type[36]), "Variable(float32, (3, 4, 5))")	# Call F.pad_sequence(xs) (line 11)
        self.assertEqual(str(id2type[37]), "[ndarray(float32, (4, 5)), ndarray(float32, (2, 5)), ndarray(float32, (2, 5))] -> Variable(float32, (3, 4, 5))")	# Attribute F.pad_sequence (line 11)
        self.assertEqual(str(id2type[41]), "[ndarray(float32, (4, 5)), ndarray(float32, (2, 5)), ndarray(float32, (2, 5))]")	# Name xs (line 11)
        self.assertEqual(str(id2type[43]), "NoneType")	# Assign
        self.assertEqual(str(id2type[44]), "Variable(float32, (3, 1, 4, 5))")	# Name xs (line 14)
        self.assertEqual(str(id2type[46]), "Variable(float32, (3, 1, 4, 5))")	# Call F.swapaxes(F.reshape(xs, (xs.shape[0], xs.shape[1], self.in_channel, xs.shape[2] // self.in_channel)), 1, 2) (line 14)
        self.assertEqual(str(id2type[47]), "Variable(float32, (3, 4, 1, 5)) -> int -> int -> Variable(float32, (3, 1, 4, 5))")	# Attribute F.swapaxes (line 14)
        self.assertEqual(str(id2type[51]), "Variable(float32, (3, 4, 1, 5))")	# Call F.reshape(xs, (xs.shape[0], xs.shape[1], self.in_channel, xs.shape[2] // self.in_channel)) (line 14)
        self.assertEqual(str(id2type[52]), "Variable(float32, (3, 4, 5)) -> (int, int, int, int) -> Variable(float32, (3, 4, 1, 5))")	# Attribute F.reshape (line 14)
        self.assertEqual(str(id2type[56]), "Variable(float32, (3, 4, 5))")	# Name xs (line 15)
        self.assertEqual(str(id2type[58]), "(int, int, int, int)")	# Tuple (xs.shape[0], xs.shape[1], self.in_channel, xs.shape[2] // self.in_channel) (line 15)
        self.assertEqual(str(id2type[59]), "int")	# Subscript xs.shape[0] (line 15)
        self.assertEqual(str(id2type[60]), "(int, int, int)")	# Attribute xs.shape (line 15)
        self.assertEqual(str(id2type[61]), "Variable(float32, (3, 4, 5))")	# Name xs (line 15)
        self.assertEqual(str(id2type[65]), "int")	# Constant 0 (line 15)
        self.assertEqual(str(id2type[67]), "int")	# Subscript xs.shape[1] (line 15)
        self.assertEqual(str(id2type[68]), "(int, int, int)")	# Attribute xs.shape (line 15)
        self.assertEqual(str(id2type[69]), "Variable(float32, (3, 4, 5))")	# Name xs (line 15)
        self.assertEqual(str(id2type[73]), "int")	# Constant 1 (line 15)
        self.assertEqual(str(id2type[75]), "int")	# Attribute self.in_channel (line 15)
        self.assertEqual(str(id2type[76]), "class VGG2L")	# Name self (line 15)
        self.assertEqual(str(id2type[79]), "int")	# BinOp xs.shape[2] // self.in_channel (line 15)
        self.assertEqual(str(id2type[80]), "int")	# Subscript xs.shape[2] (line 15)
        self.assertEqual(str(id2type[81]), "(int, int, int)")	# Attribute xs.shape (line 15)
        self.assertEqual(str(id2type[82]), "Variable(float32, (3, 4, 5))")	# Name xs (line 15)
        self.assertEqual(str(id2type[86]), "int")	# Constant 2 (line 15)
        self.assertEqual(str(id2type[88]), "int -> int -> int")	# FloorDiv
        self.assertEqual(str(id2type[89]), "int")	# Attribute self.in_channel (line 15)
        self.assertEqual(str(id2type[90]), "class VGG2L")	# Name self (line 15)
        self.assertEqual(str(id2type[94]), "int")	# Constant 1 (line 15)
        self.assertEqual(str(id2type[95]), "int")	# Constant 2 (line 15)
        self.assertEqual(str(id2type[96]), "NoneType")	# Assign
        self.assertEqual(str(id2type[97]), "Variable(float32, (3, 64, 4, 5))")	# Name xs (line 17)
        self.assertEqual(str(id2type[99]), "Variable(float32, (3, 64, 4, 5))")	# Call F.relu(self.conv1_1(xs)) (line 17)
        self.assertEqual(str(id2type[100]), "Variable(float32, (3, 64, 4, 5)) -> Variable(float32, (3, 64, 4, 5))")	# Attribute F.relu (line 17)
        self.assertEqual(str(id2type[104]), "Variable(float32, (3, 64, 4, 5))")	# Call self.conv1_1(xs) (line 17)
        self.assertEqual(str(id2type[105]), "Variable(float32, (3, 1, 4, 5)) -> Variable(float32, (3, 64, 4, 5))")	# Attribute self.conv1_1 (line 17)
        self.assertEqual(str(id2type[106]), "class VGG2L")	# Name self (line 17)
        self.assertEqual(str(id2type[109]), "Variable(float32, (3, 1, 4, 5))")	# Name xs (line 17)
        self.assertEqual(str(id2type[111]), "NoneType")	# Assign
        self.assertEqual(str(id2type[112]), "Variable(float32, (3, 64, 4, 5))")	# Name xs (line 18)
        self.assertEqual(str(id2type[114]), "Variable(float32, (3, 64, 4, 5))")	# Call F.relu(self.conv1_2(xs)) (line 18)
        self.assertEqual(str(id2type[115]), "Variable(float32, (3, 64, 4, 5)) -> Variable(float32, (3, 64, 4, 5))")	# Attribute F.relu (line 18)
        self.assertEqual(str(id2type[119]), "Variable(float32, (3, 64, 4, 5))")	# Call self.conv1_2(xs) (line 18)
        self.assertEqual(str(id2type[120]), "Variable(float32, (3, 64, 4, 5)) -> Variable(float32, (3, 64, 4, 5))")	# Attribute self.conv1_2 (line 18)
        self.assertEqual(str(id2type[121]), "class VGG2L")	# Name self (line 18)
        self.assertEqual(str(id2type[124]), "Variable(float32, (3, 64, 4, 5))")	# Name xs (line 18)
        self.assertEqual(str(id2type[126]), "NoneType")	# Assign
        self.assertEqual(str(id2type[127]), "Variable(float32, (3, 64, 2, 3))")	# Name xs (line 19)
        self.assertEqual(str(id2type[129]), "Variable(float32, (3, 64, 2, 3))")	# Call F.max_pooling_2d(xs, 2, stride=2) (line 19)
        self.assertEqual(str(id2type[130]), "Variable(float32, (3, 64, 4, 5)) -> int -> Variable(float32, (3, 64, 2, 3))")	# Attribute F.max_pooling_2d (line 19)
        self.assertEqual(str(id2type[134]), "Variable(float32, (3, 64, 4, 5))")	# Name xs (line 19)
        self.assertEqual(str(id2type[136]), "int")	# Constant 2 (line 19)
        self.assertEqual(str(id2type[138]), "int")	# Constant 2 (line 19)
        self.assertEqual(str(id2type[139]), "NoneType")	# Assign
        self.assertEqual(str(id2type[140]), "Variable(float32, (3, 128, 2, 3))")	# Name xs (line 21)
        self.assertEqual(str(id2type[142]), "Variable(float32, (3, 128, 2, 3))")	# Call F.relu(self.conv2_1(xs)) (line 21)
        self.assertEqual(str(id2type[143]), "Variable(float32, (3, 128, 2, 3)) -> Variable(float32, (3, 128, 2, 3))")	# Attribute F.relu (line 21)
        self.assertEqual(str(id2type[147]), "Variable(float32, (3, 128, 2, 3))")	# Call self.conv2_1(xs) (line 21)
        self.assertEqual(str(id2type[148]), "Variable(float32, (3, 64, 2, 3)) -> Variable(float32, (3, 128, 2, 3))")	# Attribute self.conv2_1 (line 21)
        self.assertEqual(str(id2type[149]), "class VGG2L")	# Name self (line 21)
        self.assertEqual(str(id2type[152]), "Variable(float32, (3, 64, 2, 3))")	# Name xs (line 21)
        self.assertEqual(str(id2type[154]), "NoneType")	# Assign
        self.assertEqual(str(id2type[155]), "Variable(float32, (3, 128, 2, 3))")	# Name xs (line 22)
        self.assertEqual(str(id2type[157]), "Variable(float32, (3, 128, 2, 3))")	# Call F.relu(self.conv2_2(xs)) (line 22)
        self.assertEqual(str(id2type[158]), "Variable(float32, (3, 128, 2, 3)) -> Variable(float32, (3, 128, 2, 3))")	# Attribute F.relu (line 22)
        self.assertEqual(str(id2type[162]), "Variable(float32, (3, 128, 2, 3))")	# Call self.conv2_2(xs) (line 22)
        self.assertEqual(str(id2type[163]), "Variable(float32, (3, 128, 2, 3)) -> Variable(float32, (3, 128, 2, 3))")	# Attribute self.conv2_2 (line 22)
        self.assertEqual(str(id2type[164]), "class VGG2L")	# Name self (line 22)
        self.assertEqual(str(id2type[167]), "Variable(float32, (3, 128, 2, 3))")	# Name xs (line 22)
        self.assertEqual(str(id2type[169]), "NoneType")	# Assign
        self.assertEqual(str(id2type[170]), "Variable(float32, (3, 128, 1, 2))")	# Name xs (line 23)
        self.assertEqual(str(id2type[172]), "Variable(float32, (3, 128, 1, 2))")	# Call F.max_pooling_2d(xs, 2, stride=2) (line 23)
        self.assertEqual(str(id2type[173]), "Variable(float32, (3, 128, 2, 3)) -> int -> Variable(float32, (3, 128, 1, 2))")	# Attribute F.max_pooling_2d (line 23)
        self.assertEqual(str(id2type[177]), "Variable(float32, (3, 128, 2, 3))")	# Name xs (line 23)
        self.assertEqual(str(id2type[179]), "int")	# Constant 2 (line 23)
        self.assertEqual(str(id2type[181]), "int")	# Constant 2 (line 23)
        self.assertEqual(str(id2type[182]), "NoneType")	# Assign
        self.assertEqual(str(id2type[183]), "ndarray(int64, (3,))")	# Name ilens (line 28)
        self.assertEqual(str(id2type[185]), "ndarray(int64, (3,))")	# BinOp ilens + 1 // 2 (line 28)
        self.assertEqual(str(id2type[186]), "ndarray(int64, (3,))")	# BinOp ilens + 1 (line 28)
        self.assertEqual(str(id2type[187]), "ndarray(int64, (3,))")	# Name ilens (line 28)
        self.assertEqual(str(id2type[189]), "ndarray(int64, (3,)) -> int -> ndarray(int64, (3,))")	# Add
        self.assertEqual(str(id2type[190]), "int")	# Constant 1 (line 28)
        self.assertEqual(str(id2type[191]), "ndarray(int64, (3,)) -> int -> ndarray(int64, (3,))")	# FloorDiv
        self.assertEqual(str(id2type[192]), "int")	# Constant 2 (line 28)
        self.assertEqual(str(id2type[193]), "NoneType")	# Assign
        self.assertEqual(str(id2type[194]), "ndarray(int64, (3,))")	# Name ilens (line 29)
        self.assertEqual(str(id2type[196]), "ndarray(int64, (3,))")	# BinOp ilens + 1 // 2 (line 29)
        self.assertEqual(str(id2type[197]), "ndarray(int64, (3,))")	# BinOp ilens + 1 (line 29)
        self.assertEqual(str(id2type[198]), "ndarray(int64, (3,))")	# Name ilens (line 29)
        self.assertEqual(str(id2type[200]), "ndarray(int64, (3,)) -> int -> ndarray(int64, (3,))")	# Add
        self.assertEqual(str(id2type[201]), "int")	# Constant 1 (line 29)
        self.assertEqual(str(id2type[202]), "ndarray(int64, (3,)) -> int -> ndarray(int64, (3,))")	# FloorDiv
        self.assertEqual(str(id2type[203]), "int")	# Constant 2 (line 29)
        self.assertEqual(str(id2type[204]), "NoneType")	# Assign
        self.assertEqual(str(id2type[205]), "Variable(float32, (3, 1, 128, 2))")	# Name xs (line 36)
        self.assertEqual(str(id2type[207]), "Variable(float32, (3, 1, 128, 2))")	# Call F.swapaxes(xs, 1, 2) (line 36)
        self.assertEqual(str(id2type[208]), "Variable(float32, (3, 128, 1, 2)) -> int -> int -> Variable(float32, (3, 1, 128, 2))")	# Attribute F.swapaxes (line 36)
        self.assertEqual(str(id2type[212]), "Variable(float32, (3, 128, 1, 2))")	# Name xs (line 36)
        self.assertEqual(str(id2type[214]), "int")	# Constant 1 (line 36)
        self.assertEqual(str(id2type[215]), "int")	# Constant 2 (line 36)
        self.assertEqual(str(id2type[216]), "NoneType")	# Assign
        self.assertEqual(str(id2type[217]), "Variable(float32, (3, 1, 256))")	# Name xs (line 37)
        self.assertEqual(str(id2type[219]), "Variable(float32, (3, 1, 256))")	# Call F.reshape(xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3])) (line 37)
        self.assertEqual(str(id2type[220]), "Variable(float32, (3, 1, 128, 2)) -> (int, int, int) -> Variable(float32, (3, 1, 256))")	# Attribute F.reshape (line 37)
        self.assertEqual(str(id2type[224]), "Variable(float32, (3, 1, 128, 2))")	# Name xs (line 38)
        self.assertEqual(str(id2type[226]), "(int, int, int)")	# Tuple (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]) (line 38)
        self.assertEqual(str(id2type[227]), "int")	# Subscript xs.shape[0] (line 38)
        self.assertEqual(str(id2type[228]), "(int, int, int, int)")	# Attribute xs.shape (line 38)
        self.assertEqual(str(id2type[229]), "Variable(float32, (3, 1, 128, 2))")	# Name xs (line 38)
        self.assertEqual(str(id2type[233]), "int")	# Constant 0 (line 38)
        self.assertEqual(str(id2type[235]), "int")	# Subscript xs.shape[1] (line 38)
        self.assertEqual(str(id2type[236]), "(int, int, int, int)")	# Attribute xs.shape (line 38)
        self.assertEqual(str(id2type[237]), "Variable(float32, (3, 1, 128, 2))")	# Name xs (line 38)
        self.assertEqual(str(id2type[241]), "int")	# Constant 1 (line 38)
        self.assertEqual(str(id2type[243]), "int")	# BinOp xs.shape[2] * xs.shape[3] (line 38)
        self.assertEqual(str(id2type[244]), "int")	# Subscript xs.shape[2] (line 38)
        self.assertEqual(str(id2type[245]), "(int, int, int, int)")	# Attribute xs.shape (line 38)
        self.assertEqual(str(id2type[246]), "Variable(float32, (3, 1, 128, 2))")	# Name xs (line 38)
        self.assertEqual(str(id2type[250]), "int")	# Constant 2 (line 38)
        self.assertEqual(str(id2type[252]), "int -> int -> int")	# Mult
        self.assertEqual(str(id2type[253]), "int")	# Subscript xs.shape[3] (line 38)
        self.assertEqual(str(id2type[254]), "(int, int, int, int)")	# Attribute xs.shape (line 38)
        self.assertEqual(str(id2type[255]), "Variable(float32, (3, 1, 128, 2))")	# Name xs (line 38)
        self.assertEqual(str(id2type[259]), "int")	# Constant 3 (line 38)
        self.assertEqual(str(id2type[262]), "NoneType")	# Assign
        self.assertEqual(str(id2type[263]), "Variable(float32, (None, 256)) list")	# Name xs (line 39)
        self.assertEqual(str(id2type[265]), "Variable(float32, (None, 256)) list")	# ListComp  (line 39)
        self.assertEqual(str(id2type[266]), "Variable(float32, (None, 256))")	# Subscript xs[i, :ilens[i]:, ::] (line 39)
        self.assertEqual(str(id2type[267]), "Variable(float32, (3, 1, 256))")	# Name xs (line 39)
        self.assertEqual(str(id2type[271]), "int")	# Name i (line 39)
        self.assertEqual(str(id2type[274]), "ndarray(int64, ())")	# Subscript ilens[i] (line 39)
        self.assertEqual(str(id2type[275]), "ndarray(int64, (3,))")	# Name ilens (line 39)
        self.assertEqual(str(id2type[278]), "int")	# Name i (line 39)
        self.assertEqual(str(id2type[284]), "int")	# Name i (line 39)
        self.assertEqual(str(id2type[286]), "int list")	# Call range(len(ilens)) (line 39)
        self.assertEqual(str(id2type[287]), "int -> int list")	# Name range (line 39)
        self.assertEqual(str(id2type[289]), "int")	# Call len(ilens) (line 39)
        self.assertEqual(str(id2type[290]), "ndarray(int64, (3,)) -> int")	# Name len (line 39)
        self.assertEqual(str(id2type[292]), "ndarray(int64, (3,))")	# Name ilens (line 39)
        self.assertEqual(str(id2type[294]), "(Variable(float32, (None, 256)) list, ndarray(int64, (3,)))")	# Return
        self.assertEqual(str(id2type[295]), "(Variable(float32, (None, 256)) list, ndarray(int64, (3,)))")	# Tuple (xs, ilens) (line 41)
        self.assertEqual(str(id2type[296]), "Variable(float32, (None, 256)) list")	# Name xs (line 41)
        self.assertEqual(str(id2type[298]), "ndarray(int64, (3,))")	# Name ilens (line 41)
        # === END ASSERTIONS for VGG2L ===


    def test_BLSTM(self):
        model, forward_args = gen_BLSTM_model()
        id2type = generate_id2type_from_forward(model, forward_args)

        # === BEGIN ASSERTIONS for BLSTM ===
        self.assertEqual(str(id2type[1]), "class BLSTM -> [ndarray(float32, (4, 5)), ndarray(float32, (2, 5)), ndarray(float32, (2, 5))] -> ndarray(int64, (3,)) -> ((Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7))), ndarray(int64, (3,)))")	# FunctionDef forward (line 1)
        self.assertEqual(str(id2type[9]), "NoneType")	# Expr
        self.assertEqual(str(id2type[10]), "string")	# Constant "..." (line 7)
        self.assertEqual(str(id2type[11]), "NoneType")	# Expr
        self.assertEqual(str(id2type[12]), "NoneType")	# Call logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens)) (line 8)
        self.assertEqual(str(id2type[33]), "NoneType")	# Assign
        self.assertEqual(str(id2type[34]), "ndarray(int64, (3,))")	# Name ilens (line 10)
        self.assertEqual(str(id2type[36]), "ndarray(int64, (3,))")	# Call cuda.to_cpu(ilens) (line 10)
        self.assertEqual(str(id2type[37]), "ndarray(int64, (3,)) -> ndarray(int64, (3,))")	# Attribute cuda.to_cpu (line 10)
        self.assertEqual(str(id2type[41]), "ndarray(int64, (3,))")	# Name ilens (line 10)
        self.assertEqual(str(id2type[43]), "NoneType")	# Assign
        self.assertEqual(str(id2type[44]), "(Variable(float32, (4, 3, 3)), Variable(float32, (4, 3, 3)), [Variable(float32, (4, 6)), Variable(float32, (2, 6)), Variable(float32, (2, 6))])")	# Tuple (hy, cy, ys) (line 11)
        self.assertEqual(str(id2type[45]), "Variable(float32, (4, 3, 3))")	# Name hy (line 11)
        self.assertEqual(str(id2type[47]), "Variable(float32, (4, 3, 3))")	# Name cy (line 11)
        self.assertEqual(str(id2type[49]), "[Variable(float32, (4, 6)), Variable(float32, (2, 6)), Variable(float32, (2, 6))]")	# Name ys (line 11)
        self.assertEqual(str(id2type[52]), "(Variable(float32, (4, 3, 3)), Variable(float32, (4, 3, 3)), [Variable(float32, (4, 6)), Variable(float32, (2, 6)), Variable(float32, (2, 6))])")	# Call self.nblstm(None, None, xs) (line 11)
        self.assertEqual(str(id2type[53]), "NoneType -> NoneType -> [ndarray(float32, (4, 5)), ndarray(float32, (2, 5)), ndarray(float32, (2, 5))] -> (Variable(float32, (4, 3, 3)), Variable(float32, (4, 3, 3)), [Variable(float32, (4, 6)), Variable(float32, (2, 6)), Variable(float32, (2, 6))])")	# Attribute self.nblstm (line 11)
        self.assertEqual(str(id2type[54]), "class BLSTM")	# Name self (line 11)
        self.assertEqual(str(id2type[57]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[58]), "NoneType")	# Constant None (line 11)
        self.assertEqual(str(id2type[59]), "[ndarray(float32, (4, 5)), ndarray(float32, (2, 5)), ndarray(float32, (2, 5))]")	# Name xs (line 11)
        self.assertEqual(str(id2type[61]), "NoneType")	# Assign
        self.assertEqual(str(id2type[62]), "Variable(float32, (8, 7))")	# Name ys (line 12)
        self.assertEqual(str(id2type[64]), "Variable(float32, (8, 7))")	# Call self.l_last(F.vstack(ys)) (line 12)
        self.assertEqual(str(id2type[65]), "Variable(float32, (8, 6)) -> Variable(float32, (8, 7))")	# Attribute self.l_last (line 12)
        self.assertEqual(str(id2type[66]), "class BLSTM")	# Name self (line 12)
        self.assertEqual(str(id2type[69]), "Variable(float32, (8, 6))")	# Call F.vstack(ys) (line 12)
        self.assertEqual(str(id2type[70]), "[Variable(float32, (4, 6)), Variable(float32, (2, 6)), Variable(float32, (2, 6))] -> Variable(float32, (8, 6))")	# Attribute F.vstack (line 12)
        self.assertEqual(str(id2type[74]), "[Variable(float32, (4, 6)), Variable(float32, (2, 6)), Variable(float32, (2, 6))]")	# Name ys (line 12)
        self.assertEqual(str(id2type[76]), "NoneType")	# Assign
        self.assertEqual(str(id2type[77]), "(Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Name xs (line 13)
        self.assertEqual(str(id2type[79]), "(Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Call F.split_axis(ys, np.cumsum(ilens[:-1:]), 0) (line 13)
        self.assertEqual(str(id2type[80]), "Variable(float32, (8, 7)) -> ndarray(int64, (2,)) -> int -> (Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Attribute F.split_axis (line 13)
        self.assertEqual(str(id2type[84]), "Variable(float32, (8, 7))")	# Name ys (line 13)
        self.assertEqual(str(id2type[86]), "ndarray(int64, (2,))")	# Call np.cumsum(ilens[:-1:]) (line 13)
        self.assertEqual(str(id2type[87]), "ndarray(int64, (2,)) -> ndarray(int64, (2,))")	# Attribute np.cumsum (line 13)
        self.assertEqual(str(id2type[91]), "ndarray(int64, (2,))")	# Subscript ilens[:-1:] (line 13)
        self.assertEqual(str(id2type[92]), "ndarray(int64, (3,))")	# Name ilens (line 13)
        self.assertEqual(str(id2type[95]), "int")	# UnaryOp -1 (line 13)
        self.assertEqual(str(id2type[97]), "int")	# Constant 1 (line 13)
        self.assertEqual(str(id2type[99]), "int")	# Constant 0 (line 13)
        self.assertEqual(str(id2type[100]), "NoneType")	# Delete
        self.assertEqual(str(id2type[105]), "NoneType")	# Assign
        self.assertEqual(str(id2type[106]), "(Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Name xs (line 17)
        self.assertEqual(str(id2type[108]), "(Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Call F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1:]), 0) (line 17)
        self.assertEqual(str(id2type[109]), "Variable(float32, (None, 7)) -> ndarray(int64, (2,)) -> int -> (Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Attribute F.split_axis (line 17)
        self.assertEqual(str(id2type[113]), "Variable(float32, (None, 7))")	# Call F.tanh(F.vstack(xs)) (line 17)
        self.assertEqual(str(id2type[114]), "Variable(float32, (None, 7)) -> Variable(float32, (None, 7))")	# Attribute F.tanh (line 17)
        self.assertEqual(str(id2type[118]), "Variable(float32, (None, 7))")	# Call F.vstack(xs) (line 17)
        self.assertEqual(str(id2type[119]), "(Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7))) -> Variable(float32, (None, 7))")	# Attribute F.vstack (line 17)
        self.assertEqual(str(id2type[123]), "(Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Name xs (line 17)
        self.assertEqual(str(id2type[125]), "ndarray(int64, (2,))")	# Call np.cumsum(ilens[:-1:]) (line 17)
        self.assertEqual(str(id2type[126]), "ndarray(int64, (2,)) -> ndarray(int64, (2,))")	# Attribute np.cumsum (line 17)
        self.assertEqual(str(id2type[130]), "ndarray(int64, (2,))")	# Subscript ilens[:-1:] (line 17)
        self.assertEqual(str(id2type[131]), "ndarray(int64, (3,))")	# Name ilens (line 17)
        self.assertEqual(str(id2type[134]), "int")	# UnaryOp -1 (line 17)
        self.assertEqual(str(id2type[136]), "int")	# Constant 1 (line 17)
        self.assertEqual(str(id2type[138]), "int")	# Constant 0 (line 17)
        self.assertEqual(str(id2type[139]), "((Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7))), ndarray(int64, (3,)))")	# Return
        self.assertEqual(str(id2type[140]), "((Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7))), ndarray(int64, (3,)))")	# Tuple (xs, ilens) (line 24)
        self.assertEqual(str(id2type[141]), "(Variable(float32, (None, 7)), Variable(float32, (None, 7)), Variable(float32, (None, 7)))")	# Name xs (line 24)
        self.assertEqual(str(id2type[143]), "ndarray(int64, (3,))")	# Name ilens (line 24)
        # === END ASSERTIONS for BLSTM ===


def main():
    unittest.main()

if __name__ == '__main__':
    main()
