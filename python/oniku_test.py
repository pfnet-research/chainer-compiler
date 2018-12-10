import os
import pytest
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainerx.testing
import numpy as np
import cupy

oniku_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(oniku_root, 'ch2o'))
sys.path.append(os.path.join(oniku_root, 'python'))
sys.path.append(os.path.join(oniku_root, 'build/python'))

import oniku


def aranges(xp, *shape):
    r = np.prod(shape)
    return np.arange(r).reshape(shape).astype(np.float32)


def test_flatten():
    flat = [np.array(x) for x in [0, 1, 2, 3, 4]]
    nested = [flat[0], [flat[1]], [(flat[2], [flat[3], flat[4]])]]
    assert flat == oniku._flatten(nested)


def test_unflatten():
    flat = [np.array(x) for x in [0, 1, 2, 3, 4]]
    expected = [flat[0], [flat[1]], [(flat[2], [flat[3], flat[4]])]]
    zero = np.array(0)
    tmpl = [zero, [zero], [(zero, [[zero], zero])]]
    nested, i = oniku._unflatten(flat, tmpl)
    assert expected == nested
    assert i == len(flat)


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Sequence(chainer.Chain):

    def forward(self, xs):
        s = xs[0]
        ys = [s]
        for x in xs[1:]:
            s = F.relu(s) * x
            ys.append(s)
        return ys


class MultiInOuts(chainer.Chain):

    def forward(self, x, y):
        return x + y, x * y


class SequenceGrad(chainer.Chain):

    def __init__(self, n_units):
        super(SequenceGrad, self).__init__()
        with self.init_scope():
            self.l = L.Linear(n_units, n_units)

    def forward(self, xs):
        s = xs[0]
        ys = [chainer.Variable(s)]
        for x in xs[1:]:
            s = self.l(s) + x
            ys.append(s)
        return ys


# TODO(hamaji): Figure out why this is necessary.
def _accuracy(y, t):
    y = chainer.backend.from_chainerx(y.array)
    result = chainer.functions.evaluation.accuracy.accuracy(y, t)
    return result


def _assert_allclose(e, a, **kwargs):
    if isinstance(e, cupy.ndarray):
        e = chainer.cuda.to_cpu(e)
        a = chainer.cuda.to_cpu(a)
    return chainerx.testing.assert_allclose(e, a, **kwargs)



def _array(v):
    if isinstance(v, chainer.Variable):
        return v.array
    return v


def _get_device(v):
    return chainer.backend.get_device_from_array(_array(v))


@pytest.mark.parametrize('device_name', [np, (cupy, 0), 'native:0', 'cuda:0'])
def test_mnist(device_name):
    np.random.seed(40)
    cupy.random.seed(40)

    batch_size = 3
    in_size = 5
    n_units = 4
    n_out = 10

    device = chainer.get_device(device_name)
    device.use()

    mlp = MLP(n_units, n_out)
    model = L.Classifier(mlp)
    model.to_device(device)

    input = np.random.rand(batch_size, in_size).astype(np.float32)
    input = device.xp.array(input)
    target = device.xp.array(np.random.randint(n_out, size=batch_size))

    def run_model(model):
        model.cleargrads()
        loss = model(input, target)
        loss.grad = device.xp.ones(loss.shape, loss.dtype)
        loss.backward()
        grads = []
        for name, param in sorted(model.namedparams()):
            grads.append((name, chainer.backend.to_chainerx(param.grad)))
        loss = chainer.backend.to_chainerx(loss.array)
        return loss, grads

    expected_loss, expected_grads = run_model(model)

    mlp_compiled = oniku.compile(mlp, [input])
    model = L.Classifier(mlp_compiled, accfun=_accuracy)
    model.to_device(device)

    actual_loss, actual_grads = run_model(model)

    chainerx.testing.assert_allclose(expected_loss, actual_loss)

    assert len(expected_grads) == len(actual_grads)
    for (e_name, e_grad), (a_name, a_grad) in zip(
            expected_grads, actual_grads):
        assert e_name == a_name
        chainerx.testing.assert_allclose(e_grad, a_grad, rtol=1e-4)


@pytest.mark.parametrize('device_name', [np, (cupy, 0), 'native:0', 'cuda:0'])
def test_multi_in_outs(device_name):
    device = chainer.get_device(device_name)
    device.use()

    model = MultiInOuts()
    model.to_device(device)

    inputs = [np.array(3, dtype=np.float32), np.array(39, dtype=np.float32)]

    expected = model(*inputs)

    model = oniku.compile(model, inputs)
    model.to_device(device)

    actual = model(*inputs)

    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        e = _array(e)
        a = _array(a)
        assert _get_device(e) == _get_device(a)
        _assert_allclose(e, a)


@pytest.mark.parametrize('device_name', [np, (cupy, 0), 'native:0', 'cuda:0'])
def test_sequence(device_name):
    device = chainer.get_device(device_name)
    device.use()

    model = Sequence()
    model.to_device(device)

    xs = [device.xp.array(i + 1, dtype=np.float32) for i in range(3)]
    expected = model(xs)

    model = oniku.compile(model, [xs])
    model.to_device(device)

    xs = [device.xp.array(i + 1, dtype=np.float32) for i in range(3)]
    actual = model(xs)

    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        e = _array(e)
        a = _array(a)
        assert _get_device(e) == _get_device(a)
        _assert_allclose(e, a)


@pytest.mark.parametrize('device_name', [np, (cupy, 0), 'native:0', 'cuda:0'])
def test_sequence_grad(device_name):
    device = chainer.get_device(device_name)
    device.use()

    seq_length = 4
    batch_size = 2
    n_units = 3
    model = SequenceGrad(n_units)
    model.to_device(device)

    xs = aranges(device.xp, seq_length, batch_size, n_units)
    xs = [device.xp.array(x) for x in xs]

    def run_model(model):
        model.cleargrads()
        ys = model(xs)
        loss = F.sum(F.stack(ys))
        loss.grad = device.xp.ones(loss.shape, loss.dtype)
        loss.backward()
        grads = []
        for name, param in sorted(model.namedparams()):
            grads.append((name, chainer.backend.to_chainerx(param.grad)))
        loss = chainer.backend.to_chainerx(loss.array)
        return ys, grads

    expected_ys, expected_grads = run_model(model)

    model = oniku.compile(model, [xs])
    model.to_device(device)
    actual_ys, actual_grads = run_model(model)

    assert len(expected_ys) == len(actual_ys)
    for e, a in zip(expected_ys, actual_ys):
        e = _array(e)
        a = _array(a)
        assert _get_device(e) == _get_device(a)
        _assert_allclose(e, a, rtol=1e-4)

    assert len(expected_grads) == len(actual_grads)
    for (e_name, e_grad), (a_name, a_grad) in zip(
            expected_grads, actual_grads):
        assert e_name == a_name
        _assert_allclose(e_grad, a_grad, rtol=1e-4)
