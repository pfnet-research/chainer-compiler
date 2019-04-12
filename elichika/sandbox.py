import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import os

import elichika
import elichika.parser as parser
import elichika.parser.visualizer as visualizer

class A(chainer.Chain):
    def forward(self, xs, xss, ps):
        y1 = xs[2]
        y2 = xs[3:5]
        y3 = xs[ps[0]]
        y4 = xs[ps[0]:ps[0]+2]
        y5 = xss[ps[0]:10, ps[1]:ps[1]+4]
        y6 = xss[ps[0], ps[1]:ps[1]+4]
        y7 = xss[3, ps[0]]
        y8 = xs[-1]
        y9 = xs[-2]
        # TODO(satos) listによるインデクシングもできるようにする
        # y10 = xs[[1,3,5]]
        return y1, y2, y3, y4, y5, y6, y7, y8, y9


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

    n_maxlen = 10

    model = A()

    u = np.random.rand(n_maxlen+6).astype(np.float32)
    v = np.random.rand(n_maxlen+6, n_maxlen+6).astype(np.float32)
    w = np.random.randint(0, n_maxlen, size=2)

    x = np.random.rand(7, 5, 3, 4)
    export(ListSlice(), [x], 'result/SoftmaxAxis')

