
from graphviz import Digraph
from elichika.parser.core import convert_model, Graph
from elichika.parser.nodes import Node

def get_valids(list_):
    return [l for l in list_ if l is not None]

node_id = 0
node2id = {}
node_ref_count = {}

value_id = 0
value2id = {}

graph_id = 0

def reset():
    global node_id
    global node2id
    global node_ref_count

    global value_id
    global value2id

    global graph_id

    node_id = 0
    node2id = {}
    node_ref_count = {}

    value_id = 0
    value2id = {}

    graph_id = 0

def assign_id(graph : 'Graph'):
    global node_id
    global node2id
    global value_id
    global value2id

    for node in graph.nodes:
        node2id[node] = 'node_' + str(node_id)
        node_id += 1

        for input in node.inputs:
            value2id[input] = 'value_' + str(value_id)
            value_id += 1

        for subgraph in node.subgraphs:
            assign_id(subgraph)

def count_ref(graph : 'Graph'):
    global node_ref_count

    for node in graph.nodes:
        for input in get_valids(node.inputs):
            if input.generator is not None:
                if input.generator in node_ref_count:
                    node_ref_count[input.generator] += 1
                else:
                    node_ref_count[input.generator] = 1

        for subgraph in node.subgraphs:
            count_ref(subgraph)

def visit_edge(parent_dot, graph : 'Graph', is_unused_node_ignored):
    global graph_id

    with parent_dot.subgraph(name='cluster_' + str(graph_id)) as dot:
        graph_id += 1
        dot.attr(label=graph.name)

        for node in graph.nodes:

            # ignore
            if is_unused_node_ignored:
                if len(node.inputs) == 0 and  not (node in node_ref_count):
                    continue

            dot.node(node2id[node],str(node))

            for input in node.inputs:
                if str(input) != "":
                    dot.edge(value2id[input], node2id[node])

                    if input.generator is not None:
                        dot.edge(node2id[input.generator], value2id[input])
                else:
                    if input.generator is not None:
                        dot.edge(node2id[input.generator], node2id[node])

            for subgraph in node.subgraphs:
                visit_edge(parent_dot, subgraph, is_unused_node_ignored)

def visualize(path : 'str', graph : 'Graph', is_unused_node_ignored = True):
    global node_id
    global node2id
    global value_id
    global value2id
    global graph_id

    reset()

    dot = Digraph(comment='Graph')

    graph_id = 0
    node_id = 0
    node2id = {}

    assign_id(graph)

    count_ref(graph)

    for k, v in value2id.items():
        if str(k) != "":
            dot.node(v, str(k), shape='diamond')

    visit_edge(dot, graph, is_unused_node_ignored)

    dot.render(path)

