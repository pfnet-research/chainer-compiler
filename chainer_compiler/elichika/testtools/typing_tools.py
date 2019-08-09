import ast, gast
import pprint

from chainer_compiler.elichika.parser import typing
from chainer_compiler.elichika.parser import utils


class IDAssignor(gast.NodeVisitor):
    def __init__(self):
        self.counter = 0
        self.node_ids = {}

    def visit(self, node):
        self.node_ids[node] = self.counter
        self.counter += 1
        return super().visit(node)

    def run(self, node):
        self.visit(node)
        return self.node_ids


def generate_id_table(tree):
    a = IDAssignor()
    return a.run(tree)


def generate_type_table(tree, is_debug=False):
    a = IDAssignor()
    tc = typing.TypeChecker(is_debug=is_debug)
    node_ids = a.run(tree)
    node_type = tc.infer(tree)
    new_nodetype = {}
    for n, t in node_type.items():
        new_nodetype[node_ids[n]] = t

    return new_nodetype


def generate_assertion(tree):
    # print("assert isinstance({}, {})".format(
    pass


def main():
    code = utils.clip_head("""
    def forward():
        x = 0
        for i in range(2):
            x = float(i) + 1
        return x
    """)
    node = gast.ast_to_gast(ast.parse(code))
    pprint.pprint(generate_id_table(node))
    pprint.pprint(generate_type_table(node, True))


if __name__ == '__main__':
    main()
