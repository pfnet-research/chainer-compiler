import ast, gast
import inspect
import pprint
import sys

from chainer_compiler.elichika.parser import typing
from chainer_compiler.elichika.parser import utils


class IDAssignor(gast.NodeVisitor):
    def __init__(self):
        self.counter = 0
        self.node2id = {}

    def visit(self, node):
        self.node2id[node] = self.counter
        self.counter += 1
        return super().visit(node)

    def run(self, node):
        self.visit(node)
        return self.node2id


def generate_node2id(tree):
    a = IDAssignor()
    node2id = a.run(tree)
    return node2id


def generate_id2node(node2id):
    id2node = {}
    for n, i in node2id.items():
        id2node[i] = n

    return id2node


def generate_id2type(tree, args, is_debug=False, module=None):
    node2id = generate_node2id(tree)

    tc = typing.TypeChecker(is_debug=is_debug, module=module)
    func_body = tree.body[0]  # XXX: only checks first function

    node2type = tc.infer_function(func_body, args)
    id2type = {}
    for n, t in node2type.items():
        id2type[node2id[n]] = t

    return id2type


def generate_id2type_from_func(fn, args, is_debug=False):
    code = utils.clip_head(inspect.getsource(fn))
    tree = gast.ast_to_gast(ast.parse(code))
    module = sys.modules[fn.__module__]
    id2type = generate_id2type(tree, args, is_debug=is_debug, module=module)
    return id2type


def generate_assertion(type_table_name, id2type, id2node):
    for i, t in sorted(id2type.items()):
        node = id2node[i]
        comment = "\t# {}".format(type(node).__name__)
        if hasattr(node, 'lineno'):
            comment += " (line {})".format(node.lineno)

        print("self.assertEqual(str({}[{}]), \"{}\"){}".format( \
            type_table_name, i, t, comment))


import numpy as np


def main():
    def forward(self):
        x = np.array([1,2,3])
        return x[0:2]

    code = utils.clip_head(inspect.getsource(forward))
    node = gast.ast_to_gast(ast.parse(code))
    id2node = generate_id2node(generate_node2id(node))
    module = sys.modules[forward.__module__]
    id2type = generate_id2type(node, (), is_debug=True, module=module)

    # pprint.pprint(id2type)
    generate_assertion("node_type", id2type, id2node)


if __name__ == '__main__':
    main()
