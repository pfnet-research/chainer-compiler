import ast, gast
import inspect
import numpy as np
import sys
import typing

from chainer_compiler.elichika.typing import types
from chainer_compiler.elichika.typing.type_inference import InferenceEngine
from chainer_compiler.elichika.typing.utils import node_description
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

        for ns in subroutine_node.values():
            for n in ns:
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


def generate_node2type(tree, args, is_debug=False, module=None, type_hints={}):
    reset_state()
    tc = InferenceEngine(is_debug=is_debug, module=module)
    func_body = tree.body[0]  # XXX: only checks first function
    try:
        node2type = tc.infer_function_value_args(func_body, args, type_hints=type_hints)
        return node2type, tc.subroutine_node
    except Exception as e:
        tc.dump_tyenv()
        raise e


def generate_id2type(node2type, node2id):
    id2type = {}
    for n, t in node2type.items():
        if n not in node2id.keys(): continue  # user-defined modules in nn.Sequential
        id2type[node2id[n]] = t

    return id2type


def generate_assertion(type_table_name, id2type, id2node, ofile=None):
    for i, t in sorted(id2type.items()):
        node = id2node[i]
        comment = "\t# " + node_description(node)
        output = "        self.assertEqual(str({}[{}]), \"{}\"){}".format( \
                type_table_name, i, t, comment)
        if ofile is None:
            print(output)
        else:
            ofile.write(output + '\n')


# For testing
def generate_id2type_from_forward(model, args, is_debug=False):
    code = utils.clip_head(inspect.getsource(model.forward))
    tree = gast.ast_to_gast(ast.parse(code))
    module = sys.modules[model.forward.__module__]
    node2type, subroutine_node = generate_node2type(
            tree, (model,) + args, is_debug=is_debug, module=module,
            type_hints=typing.get_type_hints(model.forward))
    node2id = generate_node2id(tree, subroutine_node)
    id2type = generate_id2type(node2type, node2id)
    return id2type


# For debug
def generate_type_inference_results(model, forward_args, is_debug=True):
    code = utils.clip_head(inspect.getsource(model.forward))
    node = gast.ast_to_gast(ast.parse(code))
    # node = Canonicalizer().visit(node)
    module = sys.modules[model.forward.__module__]
    node2type, subroutine_node = generate_node2type(
        node, (model,) + forward_args, is_debug=is_debug, module=module,
        type_hints=typing.get_type_hints(model.forward))
    node2id = generate_node2id(node, subroutine_node)
    id2type = generate_id2type(node2type, node2id)
    id2node = generate_id2node(node2id)
    return id2type, id2node


def reset_state():
    np.random.seed(42)
    types.var_counter = 0


if __name__ == '__main__':
    import argparse

    import numpy as np
    import chainer
    import chainer.functions as F
    import chainer.links as L

    from tests.elichika_typing.EspNet_test import *
    from tests.elichika_typing.Models_test import *

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="Execute the script", action="store_true")
    parser.add_argument("-o",
            help="Specify file name to output the assertions", type=str)
    args = parser.parse_args()


    class Test():
        def forward(self):
            x = np.zeros((1, 1)).astype('float32')
            y = F.pad_sequence([x], length=5)
            return y


    # model, forward_args = gen_MLP_model()
    model, forward_args = gen_GoogLeNet_model()
    # model, forward_args = gen_AttDot_model()
    # model, forward_args = gen_AttLoc_model()
    # model, forward_args = gen_BLSTM_model()
    # model, forward_args = gen_VGG2L_model()
    # model, forward_args = gen_StatelessLSTM_model()
    # model, forward_args = gen_Decoder_model()
    # model, forward_args = gen_E2E_model()

    # model, forward_args = Test(), ()

    if args.e:
        model.forward(*forward_args)
    else:
        id2type, id2node = generate_type_inference_results(model, forward_args)

        if args.o:
            ofile = open(args.o, 'w')
            generate_assertion("id2type", id2type, id2node, ofile)
