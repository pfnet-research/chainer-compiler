from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser import veval_bin

import numpy as np

def infer_type(value_list):
    assert(len(value_list))

    if len(value_list) == 1:
        return value_list[0].dtype

    else:
        for v in veval_bin.binop_type_table:

            left_dtype = value_list[0].dtype
            right_dtype = infer_type(value_list[1:])

            if v[0] == nodes.BinOpType.Add and v[1] == left_dtype and v[2] == right_dtype:
                return v[3]

        assert(False), "Incompatible types {} and {} for aggregation operation".format(left_dtype, right_dtype)

def veval(op: 'nodes.AggregateOpType', values_list: 'list[values.Value]', lineprop : 'utils.LineProperty'):

    if not veval_bin.is_initialized:
        veval_bin.initialize_lazy()

    if op == nodes.AggregateOpType.And:
        bool_value = True
        for value_ in values_list:
            bool_value = bool_value and value_.internal_value
        return values.BoolValue(bool_value)

    elif op == nodes.AggregateOpType.Or:
        bool_value = False
        for value_ in values_list:
            bool_value = bool_value or value_.internal_value
        return values.BoolValue(bool_value)

    elif op != nodes.AggregateOpType.Unknown:
        if values_list:
            return_type = infer_type(values_list)
            return functions.generate_value_with_type(values_list[0], type_ = return_type)
        else:
            return values.NumberValue(None)
    
    else:
        return functions.generate_value_with_type(values_list[0], type_ = None)
