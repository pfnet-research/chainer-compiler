from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions

def veval(op : 'nodes.UnaryOpType', value : 'values.Value'):
    
    if isinstance(value, values.NumberValue):
        return functions.generate_value_with_same_type(value)

    if isinstance(value, values.BoolValue):
        return functions.generate_value_with_same_type(value)

    if op == nodes.UnaryOpType.Not and isinstance(value, values.ListValue):
        return values.BoolValue(None)

    return values.Value()
