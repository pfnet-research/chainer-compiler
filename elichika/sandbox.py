import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import os

import elichika
import elichika.parser as parser
import elichika.parser.visualizer as visualizer



class SoftmaxAxis(chainer.Chain):
    def forward(self, x):
        return F.softmax(x, axis=2)

class Basic1(chainer.Chain):
    def forward(self):
        x = 0
        for i in range(2):
            x = i + 1
        return x

class C(chainer.Chain):
    def forward(self):
        bs = 0
        cs = 0
        for i in range(4):
            cs = i
            bs = cs
        return bs, cs


class ListSlice(chainer.Chain):
    def forward(self, x):
        # Use `shape` to make a sequence.
        xs = x.shape
        y1 = np.array(xs[2])
        y2 = np.array(xs[-2])
        y3 = np.array(xs[:2])
        y4 = np.array(xs[1:3])
        y5 = np.array(xs[1::2])
        return y1, y2, y3, y4, y5

class AvgPool(chainer.Chain):

    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        y1 = F.average_pooling_2d(x, 1, stride=2)
        return y1

class ArrayCast(chainer.Chain):
    def forward(self):
        y1 = np.array([4.0, 2.0, 3.0], dtype=np.int32)
        return y1

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.BatchNormalization(3)

    def forward(self, x):
        r = self.l1(x)
        return r

def print_graph(graph : 'core.Graph'):
    for node in graph.nodes:
        print(node)

        for subgraph in node.subgraphs:
            print_graph(subgraph)

def export(model, args, path):
    inputs,outputs,g = parser.core.convert_model(model, args)
    print_graph(g)

    # require graphviz
    visualizer.visualize(path, g)

    onnx_model = elichika.compile_model(model, args)
    elichika.save_model(path + '.onnx', onnx_model.model)
    elichika.save_model_as_text(path + '.txt', onnx_model.model)

if __name__ == "__main__":
    os.makedirs('result/', exist_ok=True)

    model = A()
    v = np.random.rand(2, 3, 5, 5).astype(np.float32)
    export(model, [v], 'result/BN')
