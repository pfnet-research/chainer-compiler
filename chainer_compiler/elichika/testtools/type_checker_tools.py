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

    def run(self, node, subroutine_node):
        self.visit(node)

        for n in subroutine_node.values():
            self.visit(n)

        return self.node2id


def generate_node2id(tree, subroutine_node):
    a = IDAssignor()
    node2id = a.run(tree, subroutine_node)
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
    return node2type, tc.subroutine_node


def generate_id2type(node2type, node2id):
    id2type = {}
    for n, t in node2type.items():
        id2type[node2id[n]] = t

    return id2type


def generate_id2type_from_forward(model, args, is_debug=False):
    code = utils.clip_head(inspect.getsource(model.forward))
    tree = gast.ast_to_gast(ast.parse(code))
    module = sys.modules[model.forward.__module__]
    args = (model,) + args
    node2type, subroutine_node = generate_node2type(
            tree, args, is_debug=is_debug, module=module)
    node2id = generate_node2id(tree, subroutine_node)
    id2type = generate_id2type(node2type, node2id)
    return id2type


def node_description(node):
    type_name = type(node).__name__
    if isinstance(node, gast.FunctionDef):
        return type_name + " " + node.name
    if isinstance(node, gast.Name):
        return type_name + " " + node.id
    if isinstance(node, gast.Attribute):
        return type_name + " " + node.attr
    return type_name


def generate_assertion(type_table_name, id2type, id2node):
    for i, t in sorted(id2type.items()):
        node = id2node[i]
        comment = "\t# " + node_description(node)
        if hasattr(node, 'lineno'):
            comment += " (line {})".format(node.lineno)

        print("self.assertEqual(str({}[{}]), \"{}\"){}".format( \
            type_table_name, i, t, comment))



if __name__ == '__main__':
    import numpy as np
    import chainer
    import chainer.functions as F
    import chainer.links as L

    from testcases.elichika_tests.model.MLP import MLP
    from testcases.elichika_tests.model.Resnet_with_loss import ResNet50

    class Test():
        def forward(self):
            x = np.zeros((3, 3)).astype(np.float32)
            y = np.zeros(3, dtype='int64')


    # model = ResNet50()

    # bsize = 2
    # v = np.random.rand(bsize, 3, 224, 224).astype(np.float32)
    # t = np.random.randint(1000, size=bsize).astype(np.int32)


    model = Test()
    forward_args = (model, )

    # --------------------------------------------------------------------------
    code = utils.clip_head(inspect.getsource(model.forward))
    node = gast.ast_to_gast(ast.parse(code))
    module = sys.modules[model.forward.__module__]
    node2type, subroutine_node = generate_node2type(
            node, forward_args, is_debug=True, module=module)
    node2id = generate_node2id(node, subroutine_node)
    id2type = generate_id2type(node2type, node2id)
    id2node = generate_id2node(node2id)

    generate_assertion("id2type", id2type, id2node)
