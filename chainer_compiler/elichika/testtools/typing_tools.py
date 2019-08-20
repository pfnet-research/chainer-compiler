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


def generate_id_table(tree):  # return type: Dict[Node, id]
    a = IDAssignor()
    return a.run(tree)


def generate_type_table(tree, is_debug=False):  # return type: Dict[id, type]
    a = IDAssignor()
    tc = typing.TypeChecker(is_debug=is_debug)
    node_ids = a.run(tree)
    node_type = tc.infer(tree)
    new_nodetype = {}
    for n, t in node_type.items():
        new_nodetype[node_ids[n]] = t

    return new_nodetype


def generate_lineno_table(id_table):  # return type: Dict[id, lineno]
    ret = {}
    for t, i in id_table.items():
        ret[i] = t.lineno if hasattr(t, 'lineno') else None
    return ret


def generate_assertion(type_table_name, type_table, lineno_table):
    for k, t in sorted(type_table.items()):
        lineno = lineno_table[k]
        print("self.assertEqual(str({}[{}]), \"{}\"){}".format(
            type_table_name, k, t,
            "\t# lineno: {}".format(lineno) if lineno is not None else ""
            ))


def main():
    code = utils.clip_head("""
    def forward():
        x = 0
        for i in range(2):
            x = float(i) + 1
        return x
    """)
    node = gast.ast_to_gast(ast.parse(code))
    node_type = generate_type_table(node, True)
    id_table = generate_id_table(node)
    lineno_table = generate_lineno_table(id_table)
    # pprint.pprint(node_type)
    # pprint.pprint(id_table)
    generate_assertion("node_type", node_type, lineno_table)


if __name__ == '__main__':
    main()
