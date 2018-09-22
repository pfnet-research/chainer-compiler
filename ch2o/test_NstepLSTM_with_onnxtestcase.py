# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self, n_layer, n_in, n_out):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.NStepLSTM(n_layer, n_in, n_out, 0.1)

    def forward(self, x):
        hy, cs, ys = self.l1(None, None, x)
        return hy, cs, ys


# ======================================


class LSTM_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        # LSTM Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        C_0 = str('initial_c')
        P = str('P')
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        # print(params[X].shape)
        # print(params[W].shape)
        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = params[B] if B in params else np.zeros(
                2 * number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(
                number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def h(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        [p_i, p_o, p_f] = np.split(self.P, 3)
        h_list = []
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, np.transpose(self.W)) + np.dot(H_t, np.transpose(self.R)) + np.add(
                *np.split(self.B, 2))
            i, o, f, c = np.split(gates, 4, -1)
            # print('iofc',i,o,f,c)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1], C_t  # chainerには C_t もいる

# ====================================================================


"""
ws :: n_layer * 8 (nanikore) * outsize * insize ()
n_layer が次のやつは、 4 * 7 から 4 * 4 になる
"""


def model_x_to_init_outs(model, x):
    onnxmod, input_tensors, output_tensors = chainer2onnx.chainer2onnx(
        model, model.forward)
    checker.check_model(onnxmod)

    chainer.config.train = False
    run_chainer_model(model, x, None)
    inits = edit_onnx_protobuf(onnxmod, model)
    chainer_out = run_chainer_model(model, x, None)

    return inits, chainer_out


import code
import chainer2onnx
from test_initializer import edit_onnx_protobuf
from testcasegen import run_chainer_model
from onnx import checker

import chainer.functions as F


def check_vals(chainer_out, converted):
    if isinstance(chainer_out, chainer.Variable):
        chainer_out = chainer_out.array

    if isinstance(chainer_out, tuple) or isinstance(chainer_out, list):
        # print(len(chainer_out),len(converted))
        if not(len(chainer_out) == len(converted)):
            print(chainer_out, converted)
            assert False
        for p, q in zip(chainer_out, converted):
            check_vals(p, q)
    else:
        print('shapes', chainer_out.shape, converted.shape)
        np.testing.assert_almost_equal(chainer_out, converted, decimal=5)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    # mxnetではLSTMの挙動をtestできないっぽい
    # test_at_onnx.check_compatibility(model,[v])

    # とりあえずというかんじ
    # これはなにかのtestになっているのだろうか

    layn = 7
    model = A(layn, 3, 5)

    x = [np.random.rand(4, 3).astype(np.float32) for _ in range(2)]
    x = [x]

    inits, chainer_out = model_x_to_init_outs(model, x)
    inits = dict(inits)
    # print(inits)

    # chainer .. last_h, last_c, y_series

    x = np.array(x[0])
    # print(x.shape)
    x = np.transpose(x, (1, 0, 2))
    # print(x.shape)

    hs = None
    cs = None
    for i in range(layn):
        w = inits['_l1_%d_ws0' % i]
        print(i, x.shape, w.shape)
        ret = LSTM_Helper(
            X=x, W=w, R=inits['_l1_%d_ws1' % i], B=inits['_l1_%d_bss' % i]).step()

        h = ret[1]
        c = ret[2]
        y = np.transpose(ret[0], (1, 0, 2, 3))[0]
        x = y
        if hs is None:
            hs = h
            cs = c
        else:
            hs = np.concatenate([hs, h])
            cs = np.concatenate([cs, c])
    """
    onnx :: y_series, last_h, last_C 
    """

    hs = np.array(hs)
    cs = np.array(cs)
    #cs = hs[0].step()

    y = np.transpose(y, (1, 0, 2))
    cs = (hs, cs, y)
    """
    print(x,x[0])
    ds = F.n_step_lstm(1,1.0,
        np.zeros((1,1,1)).astype(np.float32),
        np.zeros((1,1,1)).astype(np.float32),
        [[np.zeros((1,1)).astype(np.float32) for _ in range(8)]],
        [[np.zeros((1,1)).astype(np.float32) for _ in range(8)]],
        np.array(x[0]).astype(np.float32)
    )
    print(ds)
    """

    #code.InteractiveConsole({'co': chainer_out,'ono': cs}).interact()

    # print(cs)
    # print(chainer_out)
    check_vals(chainer_out, cs)
