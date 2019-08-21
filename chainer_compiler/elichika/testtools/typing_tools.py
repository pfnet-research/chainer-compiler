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


def generate_id_table(tree):  # return type: Dict[Node, id], Dict[id, Node]
    a = IDAssignor()
    id_table = a.run(tree)

    node_table = {}
    for n, i in id_table.items():
        node_table[i] = n

    return id_table, node_table


def generate_type_table(tree, args, is_debug=False):  # return type: Dict[id, type]
    a = IDAssignor()
    tc = typing.TypeChecker(is_debug=is_debug)
    func_body = tree.body[0]  # XXX: only checks first function
    node_ids = a.run(tree)
    node_type = tc.infer_function(func_body, args)
    new_nodetype = {}
    for n, t in node_type.items():
        new_nodetype[node_ids[n]] = t

    return new_nodetype


def generate_assertion(type_table_name, type_table, node_table):
    for i, t in sorted(type_table.items()):
        node = node_table[i]
        comment = "\t# {}".format(type(node).__name__) \
                + (" (line {})".format(node.lineno) if hasattr(node, 'lineno') else "")

        print("self.assertEqual(str({}[{}]), \"{}\"){}".format( \
            type_table_name, i, t, comment))


def main():
    code = utils.clip_head("""
    def forward(self, x):
        if True:
            x += 3
        else:
            x += 10.0
        return x
    """)
    node = gast.ast_to_gast(ast.parse(code))
    node_type = generate_type_table(node, (0,), True)

    id_table, node_table = generate_id_table(node)
    # pprint.pprint(node_type)
    # pprint.pprint(id_table)
    generate_assertion("node_type", node_type, node_table)


if __name__ == '__main__':
    main()
