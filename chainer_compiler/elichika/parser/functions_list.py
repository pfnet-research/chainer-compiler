from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import graphs
from chainer_compiler.elichika.parser import utils

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class AppendFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'append'
        self.args.add_arg('self', None)
        self.args.add_arg('elmnt', None)

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalContext' = None, line=-1):
        funcArgs = self.args.merge_inputs(inst, args)

        node = nodes.NodeCall(self, funcArgs, line)

        if inst.in_container:
            raise Exception('Invalid operation')
            
        old_v = inst.get_value()
        new_v = functions.generate_value_with_same_type(old_v)

        # estimate a type contained
        if old_v.has_constant_value():
            new_v.internal_value = list(old_v.internal_value)

        for v in funcArgs.inputs[1:]:
            new_v.append(v)

        # update value
        inst.revise(new_v)

        new_v.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([new_v])

        graph.add_node(node)
        return values.NoneValue()


class Assigner(values.PredefinedValueAssigner):
    def __init__(self):
        super().__init__()
        self.target_type = type(values.ListValue)

    def assign(self, target : 'Object'):

        append_func = values.Object(
            values.FuncValue(AppendFunction(), target, None))
        target.attributes.set_predefined_obj('append', append_func)

values.predefined_value_assigners.append(Assigner())