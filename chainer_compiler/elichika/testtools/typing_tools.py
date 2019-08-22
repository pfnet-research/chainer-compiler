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
import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        loss = F.softmax_cross_entropy(h3, t)
        return loss


class Test():
    def __init__(self):
        pass

    def forward(self):
        a = [1, 2.3]
        a.append(4)
        return a


def main():
    # out_n = 4
    # batch_size = 100
    # model = MLP(8, out_n)
    model = Test()
    forward = model.forward

    # v = np.random.rand(batch_size, 3).astype(np.float32)
    # w = np.random.randint(out_n, size=batch_size)
    forward_args = (model,)


    # --------------------------------------------------------------------------
    code = utils.clip_head(inspect.getsource(forward))
    node = gast.ast_to_gast(ast.parse(code))
    id2node = generate_id2node(generate_node2id(node))
    module = sys.modules[forward.__module__]
    id2type = generate_id2type(node, forward_args, is_debug=True, module=module)

    generate_assertion("node_type", id2type, id2node)


if __name__ == '__main__':
    main()
