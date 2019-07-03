from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import config
from chainer_compiler.elichika.parser import utils

import numpy as np

def veval(op: 'nodes.BinOpType', left: 'values.Value', right: 'values.Value', lineprop : 'utils.LineProperty'):

    if isinstance(left, values.ListValue):
        return functions.generate_value_with_same_type(left)

    if isinstance(left, values.TupleValue):
        return functions.generate_value_with_same_type(left)

    return functions.generate_value_with_same_type(left)
