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

    np.random.seed(314)
    a = np.random.rand(3, 5, 4).astype(np.float32)

    export(SoftmaxAxis(), [a], 'result/SoftmaxAxis')
