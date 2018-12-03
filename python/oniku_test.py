import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainerx.testing
import numpy as np

oniku_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(oniku_root, 'ch2o'))
sys.path.append(os.path.join(oniku_root, 'python'))
sys.path.append(os.path.join(oniku_root, 'build/python'))

import oniku


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


def test_run():
    batch_size = 3
    in_size = 12
    n_units = 13
    n_out = 10

    mlp = MLP(n_units, n_out)
    model = L.Classifier(mlp)
    device = chainer.get_device('native')
    model.to_device(device)
    device.use()

    input = np.random.rand(batch_size, in_size).astype(np.float32)
    target = np.random.randint(n_out, size=batch_size)
    expected = model(input, target).array

    mlp_compiled = oniku.compile(mlp, [input])
    model = L.Classifier(mlp_compiled)
    # TODO(hamaji): Investigate why accuracy does not work.
    model.compute_accuracy = False
    actual = model(input, target).array

    chainerx.testing.assert_allclose(expected, actual)
