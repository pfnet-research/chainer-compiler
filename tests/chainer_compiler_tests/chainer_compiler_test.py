import os
import pytest
import sys

import chainer
from chainer.backend import CpuDevice
import chainer.functions as F
import chainer.links as L
import chainerx.testing
import numpy as np


all_device_names = ['@numpy', 'native:0']

try:
    import cupy
    all_device_names.extend(['@cupy:0', 'cuda:0'])
    has_cupy = True
except:
    has_cupy = False

all_translators = ['ch2o']

try:
    import onnx_chainer  # noqa
    all_translators.append('onnx_chainer')
except:
    pass

all_computation_orders = [None, 'dummy', 'dummy2']

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'build/chainer_compiler_cc'))
sys.path.append(os.path.join(project_root, 'chainer_compiler'))

from chainer_compiler import chainer_compiler  # noqa


def aranges(xp, *shape):
    r = np.prod(shape)
    return np.arange(r).reshape(shape).astype(np.float32)


def test_flatten():
    flat = [np.array(x) for x in [0, 1, 2, 3, 4]]
    nested = [flat[0], [flat[1]], [(flat[2], [flat[3], flat[4]])]]
    assert flat == chainer_compiler._flatten(nested)


def test_unflatten():
    flat = [np.array(x) for x in [0, 1, 2, 3, 4]]
    expected = [flat[0], [flat[1]], [(flat[2], [flat[3], flat[4]])]]
    zero = np.array(0)
    tmpl = [zero, [zero], [(zero, [[zero], zero])]]
    nested, i = chainer_compiler._unflatten(flat, tmpl)
    assert expected == nested
    assert i == len(flat)


def _assert_allclose(e, a, **kwargs):
    if has_cupy and isinstance(e, cupy.ndarray):
        e = chainer.cuda.to_cpu(e)
        a = chainer.cuda.to_cpu(a)
    return chainerx.testing.assert_allclose(e, a, **kwargs)


def _array(v):
    if isinstance(v, chainer.Variable):
        return v.array
    return v


def _get_device(v):
    return chainer.backend.get_device_from_array(_array(v))


def _run_fwd_bwd(model, inputs):
    model.cleargrads()
    y = model(*inputs)
    if isinstance(y, (list, tuple)):
        loss = F.sum(F.stack(y))
        y = [chainer.backend.to_chx(x.array) for x in y]
    else:
        loss = y
        y = y.array
    loss.grad = model.xp.ones(loss.shape, loss.dtype)
    loss.backward()
    grads = []
    for name, param in sorted(model.namedparams()):
        name = name.replace('/mc', '')
        grads.append((name, chainer.backend.to_chx(param.grad)))
    return y, grads


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


def skip_check(device_name, translator, computation_order):
    if translator == 'onnx_chainer':
        if device_name == 'native:0' or device_name == 'cuda:0':
            return True
    if translator == 'ch2o' and computation_order is not None:
        return True
    return False


@pytest.mark.parametrize('device_name', all_device_names)
@pytest.mark.parametrize('translator', all_translators)
@pytest.mark.parametrize('computation_order', all_computation_orders)
def test_mnist(device_name, translator, computation_order):
    if skip_check(device_name, translator, computation_order):
        pytest.skip()

    np.random.seed(40)
    if has_cupy:
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
            name = name.replace('/mc', '')
            grads.append((name, chainer.backend.to_chx(param.grad)))
        loss = chainer.backend.to_chx(loss.array)
        return loss, grads

    expected_loss, expected_grads = _run_fwd_bwd(model, [input, target])

    mlp_compiled = chainer_compiler.compile(
        mlp, [input], translator=translator,
        computation_order=computation_order)
    model = L.Classifier(mlp_compiled)
    model.to_device(device)

    actual_loss, actual_grads = _run_fwd_bwd(model, [input, target])

    _assert_allclose(expected_loss, actual_loss)

    assert len(expected_grads) == len(actual_grads)
    for (e_name, e_grad), (a_name, a_grad) in zip(
            expected_grads, actual_grads):
        assert e_name == a_name
        assert e_grad is not None, e_name
        assert a_grad is not None, a_name
        chainerx.testing.assert_allclose(e_grad, a_grad, rtol=1e-4)


class BN(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(BN, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(n_units)
            self.linear = L.Linear(n_units, n_out)

    def forward(self, x):
        return self.linear(self.bn(x))


# TODO(hamaji): Enable this test again.
# @pytest.mark.parametrize('device_name', all_device_names)
# @pytest.mark.parametrize('translator', all_translators)
# @pytest.mark.parametrize('computation_order', all_computation_orders)
# def test_bn(device_name, translator, computation_order):
#     if skip_check(device_name, translator, computation_order):
#         pytest.skip()

#     np.random.seed(40)
#     if has_cupy:
#         cupy.random.seed(40)

#     batch_size = 3
#     in_size = 5
#     n_out = 10

#     device = chainer.get_device(device_name)
#     device.use()

#     bn = BN(in_size, n_out)

#     input = np.random.rand(batch_size, in_size).astype(np.float32)
#     input = device.xp.array(input)
#     target = device.xp.array(np.random.randint(n_out, size=batch_size))

#     bn_compiled = chainer_compiler.compile(
#         bn, [input], translator=translator,
#         computation_order=computation_order)
#     model = L.Classifier(bn_compiled)
#     model.to_device(device)

#     old_avg_mean = CpuDevice().send(model.predictor.mc.bn.avg_mean.copy())
#     old_avg_var = CpuDevice().send(model.predictor.mc.bn.avg_var.copy())

#     loss, grads = _run_fwd_bwd(model, [input, target])

#     new_avg_mean = CpuDevice().send(model.predictor.mc.bn.avg_mean.copy())
#     new_avg_var = CpuDevice().send(model.predictor.mc.bn.avg_var.copy())

#     # running_mean and running_var should be updated
#     assert not np.allclose(old_avg_mean, new_avg_mean)
#     assert not np.allclose(old_avg_var, new_avg_var)


class MultiInOuts(chainer.Chain):

    def forward(self, x, y):
        return x + y, x * y


@pytest.mark.parametrize('device_name', all_device_names)
@pytest.mark.parametrize('translator', ['ch2o'])
def test_multi_in_outs(device_name, translator):
    device = chainer.get_device(device_name)
    device.use()

    model = MultiInOuts()
    model.to_device(device)

    inputs = [np.array(3, dtype=np.float32), np.array(39, dtype=np.float32)]

    expected = model(*inputs)

    model = chainer_compiler.compile(model, inputs, translator=translator)
    model.to_device(device)

    actual = model(*inputs)

    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        e = _array(e)
        a = _array(a)
        assert _get_device(e) == _get_device(a)
        _assert_allclose(e, a)


class ConstMul(chainer.Chain):

    def forward(self, x):
        return x * np.array(42.0, dtype=np.float32)


@pytest.mark.parametrize('device_name', all_device_names)
@pytest.mark.parametrize('translator', ['ch2o'])
def test_const_mul(device_name, translator):
    device = chainer.get_device(device_name)
    device.use()

    # This checks if the default ChainerX device is set properly by
    # Constant op, whose result will be placed on the default device.
    model = ConstMul()
    model.to_device(device)

    inputs = [np.array(3, dtype=np.float32)]

    expected = model(*inputs)

    model = chainer_compiler.compile(model, inputs, translator=translator)
    model.to_device(device)

    actual = model(*inputs)

    e = _array(expected)
    a = _array(actual)
    assert _get_device(e) == _get_device(a)
    _assert_allclose(e, a)


class Sequence(chainer.Chain):

    def forward(self, xs):
        s = xs[0]
        ys = [s]
        for x in xs[1:]:
            s = F.relu(s) * x
            ys.append(s)
        return ys


@pytest.mark.parametrize('device_name', all_device_names)
@pytest.mark.parametrize('translator', ['ch2o'])
def test_sequence(device_name, translator):
    device = chainer.get_device(device_name)
    device.use()

    model = Sequence()
    model.to_device(device)

    xs = [device.xp.array(i + 1, dtype=np.float32) for i in range(3)]
    expected = model(xs)

    model = chainer_compiler.compile(model, [xs], translator=translator)
    model.to_device(device)

    xs = [device.xp.array(i + 1, dtype=np.float32) for i in range(3)]
    actual = model(xs)

    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        e = _array(e)
        a = _array(a)
        assert _get_device(e) == _get_device(a)
        _assert_allclose(e, a)


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


@pytest.mark.parametrize('device_name', all_device_names)
@pytest.mark.parametrize('translator', ['ch2o'])
def test_sequence_grad(device_name, translator):
    device = chainer.get_device(device_name)
    device.use()

    seq_length = 4
    batch_size = 2
    n_units = 3
    model = SequenceGrad(n_units)
    model.to_device(device)

    xs = aranges(device.xp, seq_length, batch_size, n_units)
    xs = [device.xp.array(x) for x in xs]

    expected_ys, expected_grads = _run_fwd_bwd(model, [xs])

    model = chainer_compiler.compile(model, [xs], translator=translator)
    model.to_device(device)
    actual_ys, actual_grads = _run_fwd_bwd(model, [xs])

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
        assert e_grad is not None, e_name
        assert a_grad is not None, a_name
        _assert_allclose(e_grad, a_grad, rtol=1e-4)


class PartiallyDifferentiable(chainer.Chain):

    def __init__(self, n_units):
        super(PartiallyDifferentiable, self).__init__()
        with self.init_scope():
            self.l = L.Linear(n_units, n_units)

    def forward(self, xs, indices):
        r = xs[0]
        for i in indices:
            x = xs[i]
            r = self.l(r) * x
        return r


@pytest.mark.parametrize('device_name', ['@numpy'])
@pytest.mark.parametrize('translator', ['ch2o'])
def test_partially_differentiable(device_name, translator):
    np.random.seed(40)
    device = chainer.get_device(device_name)
    device.use()

    n_units = 3
    batch_size = 2
    seq_length = 7

    xs = aranges(device.xp, seq_length, batch_size, n_units)
    xs = [chainer.Variable(device.xp.array(x)) for x in xs]
    indices = [np.array(i, dtype=np.int32) for i in [2, 3, 5, 1]]

    model = PartiallyDifferentiable(n_units)
    model.to_device(device)

    expected_loss, expected_grads = _run_fwd_bwd(model, [xs, indices])
    # expected_gxs = [x.grad for x in xs]

    xs = aranges(device.xp, seq_length, batch_size, n_units)
    xs = [chainer.Variable(device.xp.array(x)) for x in xs]

    model = chainer_compiler.compile(model, [xs, indices],
                                     translator=translator)
    model.to_device(device)
    actual_loss, actual_grads = _run_fwd_bwd(model, [xs, indices])
    # actual_gxs = [x.grad for x in xs]

    chainerx.testing.assert_allclose(expected_loss, actual_loss, rtol=1e-5)

    assert len(expected_grads) == len(actual_grads)
    for (e_name, e_grad), (a_name, a_grad) in zip(
            expected_grads, actual_grads):
        assert e_name == a_name
        assert e_grad is not None, e_name
        assert a_grad is not None, a_name
        _assert_allclose(e_grad, a_grad, rtol=1e-4)

    # TODO(hamaji): Fix this test.
    # for e, a in zip(expected_gxs, actual_gxs):
    #     assert e is not None
    #     assert a is not None
    #     _assert_allclose(e, a)
