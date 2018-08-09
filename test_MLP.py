#coding: utf-8
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# Network definition
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


"""
# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_out) 

    def forward(self, x):
        x = F.relu(self.l1(x))
         
        #print('x',x)
        #print('l1(x)',self.l1(x))
        #print('r',r)
        return x
"""

"""
  graph = helper.make_graph(
    [
        helper.make_node("Gemm", inputs=["x", "l1_W", "l1_b"], outputs=["Y"],transA=0,transB=1),
        #helper.make_node("Sigmoid", ["H1"], ["Y"]),
    ],
    "MLP",
    [
        helper.make_tensor_value_info('x' , TensorProto.FLOAT, ['batch_size','input_size']),
        helper.make_tensor_value_info('l1_W', TensorProto.FLOAT, [2,'input_size']),
        helper.make_tensor_value_info('l1_b', TensorProto.FLOAT, [2]),
     ],
    [
        helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2]),
    ],
  )
"""

#======================================from MLP

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    import unittest
    import chainer2onnx
    import test_mxnet

    model = MLP(8,2)

     #イメージは、3*3が7枚
    v = np.random.rand(5,3).astype(np.float32)
    test_mxnet.check_compatibility(model,v)

