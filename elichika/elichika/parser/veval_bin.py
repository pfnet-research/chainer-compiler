from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions

def veval(op : 'nodes.BinOpType', left : 'values.Value', right : 'values.Value'):
    
    if isinstance(left, values.NumberValue) and isinstance(right, values.NumberValue):
        return functions.generate_value_with_same_type(left)

    if isinstance(left, values.ListValue) and isinstance(right, values.ListValue):
        return functions.generate_value_with_same_type(left)

    return values.Value()
