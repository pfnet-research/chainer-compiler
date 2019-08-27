import ast, gast
import inspect
import sys

from chainer_compiler.elichika.parser.type_checker import TypeChecker
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


def generate_node2type(tree, args, is_debug=False, module=None):
    tc = TypeChecker(is_debug=is_debug, module=module)
    func_body = tree.body[0]  # XXX: only checks first function
    node2type = tc.infer_function_vargs(func_body, args)
    return node2type


def generate_id2type(node2type, node2id):
    id2type = {}
    for n, t in node2type.items():
        if n in node2id.keys():
            id2type[node2id[n]] = t

    return id2type


def generate_id2type_from_forward(model, args, is_debug=False):
    code = utils.clip_head(inspect.getsource(model.forward))
    tree = gast.ast_to_gast(ast.parse(code))
    module = sys.modules[model.forward.__module__]
    args = (model,) + args
    node2type = generate_node2type(tree, args, is_debug=is_debug, module=module)
    node2id = generate_node2id(tree)
    id2type = generate_id2type(node2type, node2id)
    return id2type


def generate_assertion(type_table_name, id2type, id2node):
    for i, t in sorted(id2type.items()):
        node = id2node[i]
        comment = "\t# {}".format(type(node).__name__)
        if hasattr(node, 'lineno'):
            comment += " (line {})".format(node.lineno)

        print("self.assertEqual(str({}[{}]), \"{}\"){}".format( \
            type_table_name, i, t, comment))



if __name__ == '__main__':
    import numpy as np
    import chainer
    import chainer.functions as F
    import chainer.links as L


    # class A():
    #     def __init__(self, x):
    #         self.x = x

    #     def f(self, x):
    #         return self.x + x

    # class Test():
    #     def __init__(self):
    #         self.a = A(1)

    #     def forward(self, x):
    #         return self.a.f(x)

    from testcases.elichika_tests.model.MLP import MLP

    def h(x, y):
        return x + y

    class Test():
        def forward(self, x):
            return h(x, 1)

    model = Test()

    forward_args = (model, 1)

    # --------------------------------------------------------------------------
    code = utils.clip_head(inspect.getsource(model.forward))
    node = gast.ast_to_gast(ast.parse(code))
    module = sys.modules[model.forward.__module__]
    node2type = generate_node2type(
            node, forward_args, is_debug=True, module=module)
    node2id = generate_node2id(node)
    id2type = generate_id2type(node2type, node2id)
    id2node = generate_id2node(node2id)

    generate_assertion("id2type", id2type, id2node)
