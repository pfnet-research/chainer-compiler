from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import config
from chainer_compiler.elichika.parser import utils

import numpy as np

# pair op, left, right and result
binop_type_table = []

is_initialized = False

def initialize_lazy():
    global is_initialized
    if is_initialized:
        return

    # generete a type table
    def add_binop_type_table(op: 'nodes.BinOpType', left_type, right_type):
        def generate_value(type_):
            return np.array(1, dtype=type_)

        left = generate_value(left_type)
        right = generate_value(right_type)

        if op == nodes.BinOpType.Add:
            result = left + right
        elif op == nodes.BinOpType.Sub:
            result = left - right
        elif op == nodes.BinOpType.Mul:
            result = left * right
        elif op == nodes.BinOpType.Div:
            result = left / right
        elif op == nodes.BinOpType.FloorDiv:
            result = left // right
        else:
            assert(False)
    
        result_type = np.array(result).dtype

        binop_type_table.append((op, left_type, right_type, result_type))

    def add_all_binop(left_type, right_type):
        add_binop_type_table(nodes.BinOpType.Add, left_type, right_type)
        add_binop_type_table(nodes.BinOpType.Sub, left_type, right_type)
        add_binop_type_table(nodes.BinOpType.Mul, left_type, right_type)
        add_binop_type_table(nodes.BinOpType.Div, left_type, right_type)
        add_binop_type_table(nodes.BinOpType.FloorDiv, left_type, right_type)

    add_all_binop(utils.dtype_float32, utils.dtype_float32)
    add_all_binop(utils.dtype_float32, utils.dtype_int)
    add_all_binop(utils.dtype_int, utils.dtype_float32)
    add_all_binop(utils.dtype_int, utils.dtype_int)
    add_all_binop(utils.dtype_float64, utils.dtype_float64)
    add_all_binop(utils.dtype_float64, utils.dtype_int)
    add_all_binop(utils.dtype_int, utils.dtype_float64)
    add_all_binop(utils.dtype_int, utils.dtype_int)
    
    is_initialized = True

def veval(op: 'nodes.BinOpType', left: 'values.Value', right: 'values.Value', lineprop : 'utils.LineProperty'):

    initialize_lazy()

    if isinstance(left, values.ListValue):
        return functions.generate_value_with_same_type(left)

    if isinstance(left, values.TupleValue):
        return functions.generate_value_with_same_type(left)

    if isinstance(left, values.StrValue):
        if not (isinstance(right, values.NumberValue) or isinstance(right, values.StrValue) or isinstance(right, values.BoolValue)
            or isinstance(right, values.TupleValue)):
            assert False

        if not(left.has_constant_value() or right.has_constant_value()):
            assert False

        if op == nodes.BinOpType.Add:
            if not isinstance(right, values.StrValue):
                assert False
            return values.StrValue(left.internal_value + right.internal_value)

        elif op == nodes.BinOpType.Mod:
            right_internal_value = right.internal_value

            if isinstance(right, values.TupleValue):
                values_ = []
                for ref in right_internal_value:
                    if not (isinstance(ref.get_value(), values.NumberValue) or isinstance(ref.get_value(), values.StrValue)
                        or isinstance(ref.get_value(), values.BoolValue)):
                        assert False
                    if not ref.get_value().has_constant_value():
                        assert False
                    values_.append(ref.get_value())

                right_internal_value = tuple(value.internal_value for value in values_)
            return values.StrValue(left.internal_value % right_internal_value)

        return values.StrValue("")

    if isinstance(right, values.NumberValue) or isinstance(right, values.TensorValue):

        if not isinstance(left, values.NumberValue) and not isinstance(left, values.TensorValue):
            utils.print_warning('Unestimated type is on left', lineprop)
            left = values.NumberValue(0.0)

        left_type = left.dtype
        right_type = right.dtype

        if not config.float_restrict:
            if left_type == utils.dtype_float64:
                left_type = utils.dtype_float32
            if right_type == utils.dtype_float64:
                right_type = utils.dtype_float32

        if left_type is None:
            left_type = np.array(0.0, np.float32).dtype

        if right_type is None:
            right_type = np.array(0.0, np.float32).dtype

        result_type = None

        for v in binop_type_table:
            if v[0] == op and v[1] == left_type and v[2] == right_type:
                result_type = v[3]
                break

        assert(result_type is not None)

        if  isinstance(right, values.TensorValue):
            return functions.generate_value_with_type(right, type_ = result_type)
        else:
            return functions.generate_value_with_type(left, type_ = result_type)

    return values.Value()
