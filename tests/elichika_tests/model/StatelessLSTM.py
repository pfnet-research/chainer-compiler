import functools
import operator

import numpy as np
import six

import chainer
from chainer.backends import cuda
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable

from chainer import functions as F


def _extract_gates(x):
    r = F.reshape(x, (len(x), x.shape[1] // 4, 4) + x.shape[2:])
    r = F.separate(r, axis=2)
    return r[0], r[1], r[2], r[3]


def lstm_forward(c_prev, x):
    a, i, f, o = _extract_gates(x)
    batch = len(x)

    a = F.tanh(a)
    i = F.sigmoid(i)
    f = F.sigmoid(f)
    o = F.sigmoid(o)

    c_next = a * i + f * c_prev
    h = o * F.tanh(c_next)
    return c_next, h


class LSTMBase(link.Chain):

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None):
        if out_size is None:
            out_size, in_size = in_size, None

        super(LSTMBase, self).__init__()
        if bias_init is None:
            bias_init = 0
        if forget_bias_init is None:
            forget_bias_init = 1
        self.state_size = out_size
        self.lateral_init = lateral_init
        self.upward_init = upward_init
        self.bias_init = bias_init
        self.forget_bias_init = forget_bias_init

        with self.init_scope():
            self.upward = linear.Linear(in_size, 4 * out_size, initialW=0)
            self.lateral = linear.Linear(out_size, 4 * out_size, initialW=0,
                                         nobias=True)
            if in_size is not None:
                self._initialize_params()

    def _initialize_params(self):
        lateral_init = initializers._get_initializer(self.lateral_init)
        upward_init = initializers._get_initializer(self.upward_init)
        bias_init = initializers._get_initializer(self.bias_init)
        forget_bias_init = initializers._get_initializer(self.forget_bias_init)

        for i in six.moves.range(0, 4 * self.state_size, self.state_size):
            lateral_init(self.lateral.W.data[i:i + self.state_size, :])
            upward_init(self.upward.W.data[i:i + self.state_size, :])

        a, i, f, o = lstm._extract_gates(
            self.upward.b.data.reshape(1, 4 * self.state_size, 1))

        bias_init(a)
        bias_init(i)
        forget_bias_init(f)
        bias_init(o)


class StatelessLSTM(LSTMBase):

    """Stateless LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, this chain holds upward and
    lateral connections as child links. This link doesn't keep cell and
    hidden states.

    Args:
        in_size (int or None): Dimension of input vectors. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.

    .. admonition:: Example

        There are several ways to make a StatelessLSTM link.

        Let a two-dimensional input array :math:`x`, a cell state array
        :math:`h`, and the output array of the previous step :math:`h` be:

        >>> x = np.zeros((1, 10), dtype=np.float32)
        >>> c = np.zeros((1, 20), dtype=np.float32)
        >>> h = np.zeros((1, 20), dtype=np.float32)

        1. Give both ``in_size`` and ``out_size`` arguments:

            >>> l = L.StatelessLSTM(10, 20)
            >>> c_new, h_new = l(c, h, x)
            >>> c_new.shape
            (1, 20)
            >>> h_new.shape
            (1, 20)

        2. Omit ``in_size`` argument or fill it with ``None``:

            The below two cases are the same.

            >>> l = L.StatelessLSTM(20)
            >>> c_new, h_new = l(c, h, x)
            >>> c_new.shape
            (1, 20)
            >>> h_new.shape
            (1, 20)

            >>> l = L.StatelessLSTM(None, 20)
            >>> c_new, h_new = l(c, h, x)
            >>> c_new.shape
            (1, 20)
            >>> h_new.shape
            (1, 20)

    """

    # EDIT(hamaji): Eager intialization.
    def initialize_params(self, in_size):
        with cuda.get_device_from_id(self._device_id):
            self.upward._initialize_params(in_size)
            self._initialize_params()

    def forward(self, c, h, x):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous time step.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
            ``c_new`` represents new cell state, and ``h_new`` is updated
            output of LSTM units.

        """
        # EDIT(hamaji): No lazy initialization.
        # if self.upward.W.data is None:
        #     in_size = x.size // x.shape[0]
        #     with cuda.get_device_from_id(self._device_id):
        #         self.upward._initialize_params(in_size)
        #         self._initialize_params()

        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            # EDIT(hamaji): Use numpy and np.float32.
            # xp = self.xp
            # with cuda.get_device_from_id(self._device_id):
            #     c = variable.Variable(
            #         xp.zeros((x.shape[0], self.state_size), dtype=x.dtype))
            c = variable.Variable(
                self.xp.zeros((x.shape[0], self.state_size), dtype=self.xp.float32))
        # EDIT(hamaji): Use lstm_forward.
        return lstm_forward(c, lstm_in)
        # return lstm.lstm(c, lstm_in)

    def original(self, c, h, x):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous time step.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
            ``c_new`` represents new cell state, and ``h_new`` is updated
            output of LSTM units.

        """
        if self.upward.W.data is None:
            in_size = x.size // x.shape[0]
            with cuda.get_device_from_id(self._device_id):
                self.upward._initialize_params(in_size)
                self._initialize_params()

        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            with cuda.get_device_from_id(self._device_id):
                c = variable.Variable(
                    xp.zeros((x.shape[0], self.state_size), dtype=x.dtype))
        return lstm.lstm(c, lstm_in)


class StatelessLSTMBackprop(chainer.Chain):

    def __init__(self, in_size, out_size=None):
        super(StatelessLSTMBackprop, self).__init__()
        with self.init_scope():
            self.l = StatelessLSTM(in_size, out_size)

    def forward(self, c, h, x):
        c, h = self.l(c, h, x)
        return c * h


from chainer_compiler.elichika import testtools


def main():
    import numpy as np
    np.random.seed(43)

    batch_size = 3
    in_size = 7
    out_size = 4

    def model_fn():
        lstm = StatelessLSTM(in_size, out_size)
        return lstm

    c = np.random.rand(batch_size, out_size).astype(np.float32)
    h = np.random.rand(batch_size, out_size).astype(np.float32)
    x = np.random.rand(batch_size, in_size).astype(np.float32)

    model = model_fn()
    # Check if our modification is valid.
    expected = model.original(c, h, x)
    actual = model.forward(c, h, x)
    for e, a in zip(expected, actual):
        assert np.allclose(e.array, a.array)

    testtools.generate_testcase(model_fn(), [c, h, x])

    # TODO (hamaji): support func
    # testtools.generate_testcase(model_fn, [c, h, x])

    def model_fn():
        lstm = StatelessLSTMBackprop(in_size, out_size)
        return lstm

    testtools.generate_testcase(model_fn(), [c, h, x], backprop=True)

    # TODO (hamaji): support func
    # testtools.generate_testcase(model_fn, [c, h, x], backprop=True)


if __name__ == '__main__':
    main()