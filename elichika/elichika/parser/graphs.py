from elichika.parser import nodes
from elichika.parser import utils

class Graph:
    def __init__(self):
        self.name = ''
        self.nodes = []

    def add_node(self, node : 'nodes.Node'):
        self.nodes.append(node)
