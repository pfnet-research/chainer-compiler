from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions


def veval(op: 'nodes.MultiaryOpType', values_list: 'values.Value'):
    if all([isinstance(value_, values.BoolValue) for value_ in values_list]):
        if op == nodes.MultiaryOpType.And:
            bool_value = True
            for value_ in values_list:
                bool_value = bool_value and value_.internal_value
            return values.BoolValue(bool_value)
        elif op == nodes.MultiaryOpType.Or:
            bool_value = False
            for value_ in values_list:
                bool_value = bool_value or value_.internal_value
            return values.BoolValue(bool_value)

        return values.BoolValue(None)

    return values.Value()