from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import utils


class Graph:
    def __init__(self):
        self.name = ''
        self.nodes = []
        self.input_values = []
        self.output_values = []
        self.root_graph = None

    def add_input_value(self, value : 'values.Value'):
        assert(value is not None)
        self.input_values.append(value)

    def add_output_value(self, value : 'values.Value'):
        assert(value is not None)
        self.output_values.append(value)

    def add_node(self, node: 'nodes.Node'):
        self.nodes.append(node)

    def add_initial_node(self, node: 'nodes.Node'):
        self.nodes = [node] + self.nodes
