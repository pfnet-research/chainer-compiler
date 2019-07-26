from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import values
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import graphs
from chainer_compiler.elichika.parser import utils

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class KeysFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'keys'
        self.args.add_arg('self', None)

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalContext' = None, line=-1):
        funcArgs = self.args.merge_inputs(inst, args)

        if inst.in_container:
            raise Exception('Invalid operation')

        keys = inst.get_value().internal_keys.values()

        vargs = []
        vargs_value = []
        for varg in keys:
            vargs.append(utils.try_get_obj(varg, 'dict_keys', utils.LineProperty()))
            vargs_value.append(utils.try_get_obj(varg, 'dict_keys', utils.LineProperty()).get_value())

        node = nodes.NodeGenerate('List', vargs_value , line)
        graph.add_node(node)

        value = values.ListValue(vargs)
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return value


class ValuesFunction(functions.FunctionBase):
    def __init__(self):
        super().__init__()
        self.name = 'values'
        self.args.add_arg('self', None)

    def vcall(self, module: 'values.Field', graph: 'graphs.Graph', inst: 'values.Object', args: 'functions.FunctionArgInput',
              option: 'vevaluator.VEvalContext' = None, line=-1):
        funcArgs = self.args.merge_inputs(inst, args)

        if inst.in_container:
            raise Exception('Invalid operation')

        key_hashes = inst.get_value().internal_keys.keys()
        attributes = inst.get_value().internal_values
        vargs = []
        vargs_ref = []
        for hash in key_hashes:
            varg = attributes.get_attribute(hash)
            if varg.has_obj():
                vargs.append(utils.try_get_obj(varg, 'dict_values', utils.LineProperty()).get_value())
                vargs_ref.append(utils.try_get_obj(varg, 'dict_values', utils.LineProperty()))
            else:
                assert(False)

        node = nodes.NodeGenerate('List', vargs , line)
        graph.add_node(node)

        value = values.ListValue(vargs_ref)
        value.name = '@F.{}.{}'.format(line, self.name)
        node.set_outputs([value])
        return value


class Assigner(values.PredefinedValueAssigner):
    def __init__(self):
        super().__init__()
        self.target_type = type(values.DictValue)

    def assign(self, target : 'Object'):

        keys_func = values.Object(
            values.FuncValue(KeysFunction(), target, None))
        target.attributes.get_attribute('keys').revise(keys_func)

        values_func = values.Object(
            values.FuncValue(ValuesFunction(), target, None))
        target.attributes.get_attribute('values').revise(values_func)

values.predefined_value_assigners.append(Assigner())