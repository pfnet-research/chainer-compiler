import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import os

import elichika
import elichika.parser as parser
import elichika.parser.visualizer as visualizer

class MultiLayerPerceptron(chainer.Chain):
    def __init__(self, n_in, n_hidden, n_out):
        super(MultiLayerPerceptron, self).__init__()
        with self.init_scope():
            self.layer1 = L.Linear(n_in, n_hidden)
            self.layer2 = L.Linear(n_hidden, n_hidden)
            self.layer3 = L.Linear(n_hidden, n_out)

    def forward(self, x):
        # Forward propagation
        self.x = x
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        return self.layer3(h2)

class DynamicCond(chainer.Chain):
    def forward(self, x, cond):
        if cond:
            x = 3
        else:
            x = 10
        return x

class D(chainer.Chain):
    def forward(self):
        for i in range(4):
            o = i
        return o

class UpdateSelf(chainer.Chain):
    def forward(self, x):
        self.x = x
        for i in range(5):
            self.x = self.x + i
        return self.x

class SimpleFunc(chainer.Chain):
    def forward(self):
        a,b = self.get_value()
        return a + b

    def get_value(self):
        return 1.0, 2.0

class Conv(chainer.Chain):

    def __init__(self):
        super(Conv, self).__init__()
        with self.init_scope():
            # TODO Add more tests
            self.l1 = L.Convolution2D(None, 6, (5, 7), stride=(2, 3))

    def forward(self, x):
        y1 = self.l1(x)
        return y1

class IsNot(chainer.Chain):
    def forward(self, x, y):
        return x is not y


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
    
if __name__ == "__main__":
    os.makedirs('result/', exist_ok=True)

    #resnet50 = chainer.links.ResNet50Layers()
    m = IsNot()
    export(m, [[42],[42]], 'result/Conv')

    #onnx_model = elichika.compile_model(m, [np.zeros((10))])
    #elichika.save_model('result/MLP_model.onnx', onnx_model.model)
    #elichika.save_model_as_text('result/MLP_model.txt', onnx_model.model)
    
    all = False
    if all:
        m = SimpleFunc()
        export(m, [], 'result/SimpleFunc')

        m = MultiLayerPerceptron(10, 10, 10)
        export(m, [np.zeros((10))], 'result/MLP')

        m = DynamicCond()
        export(m, [0.0, True], 'result/DynamicCond')

        m = UpdateSelf()
        export(m, [10.0], 'result/UpdateSelf')
