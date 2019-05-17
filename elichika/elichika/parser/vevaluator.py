import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import ast, gast

from elichika.parser import config
from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import utils
from elichika.parser.graphs import Graph
from elichika.parser import veval_bin
from elichika.parser import veval_unary

import numpy as np

def try_get_ref(value, name, lineprop) -> 'values.ValueRef':
    if value is None:
        if config.show_warnings:
            print('Failed to get object {}. in {}'.format(name, lineprop))
        return None

    if isinstance(value, values.Value):
        if config.show_warnings:
            print('Failed to get object {}. in {}. value is Value.'.format(name, lineprop))
        return None

    if isinstance(value, values.Attribute):
        if value.has_obj():
            return value.get_ref()

    if isinstance(value, values.ValueRef):
        return value

    return None

def try_get_value(value, name, lineprop, is_none_allowed = False) -> 'values.Value':
    if value is None:
        if config.show_warnings:
            print('Failed to get value {}. in {}'.format(name, lineprop))
        return None

    if isinstance(value, values.NoneValue) and not is_none_allowed:
        if config.show_warnings:
            print('Value {} is none. in {}'.format(name, lineprop))
        return None

    if isinstance(value, values.Value):
        return value

    if isinstance(value, values.ValueRef):
        return value.get_value()

    if isinstance(value, values.Attribute):
        return value.get_ref().get_value()

    raise Exception('Value {} is invalid. in L.{}'.format(name, lineprop))

def get_ast_name_forcibly(ast):
    if isinstance(ast, gast.gast.Name):
        return ast.id
    if isinstance(ast, gast.gast.Attribute):
        return ast.attr
    return ''

class AstContext:
    def __init__(self, nast, lineno_offset : 'int'):
        self.nast = nast
        self.lineno_offset = lineno_offset
        self.lineno = self.lineno_offset
        if hasattr(self.nast, 'lineno'):
            self.lineno = self.nast.lineno + self.lineno_offset

    def c(self, value) -> 'AstContext':
        """
        get AstContext including value
        """
        return AstContext(value, self.lineno_offset)

class VEvalOption:
    def __init__(self):
        self.eval_as_written_target = False

def veval_ast_attribute(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph', option : 'VEvalOption' = None) -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Attribute))
    lineprop = utils.LineProperty(astc.lineno)

    from_module = True
    if option is not None and option.eval_as_written_target:
        from_module = False

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    value_ref = try_get_ref(value, 'attribute', lineprop)
    attr = value_ref.get_field().get_attribute(astc.nast.attr, from_module)

    # property(getter)
    if attr.has_obj() and isinstance(attr.get_ref(False).get_value(), values.FuncValue) and attr.get_ref(False).get_value().func.is_property:
        func_value = attr.get_ref(False).get_value()
        ret = func_value.func.vcall(local_field.module, graph, func_value.obj, functions.FunctionArgInput(), lineprop)
        return ret

    if attr.has_obj():
        return attr

    # if attr is not found
    gotten_obj = value_ref.try_get_and_store_obj(astc.nast.attr, graph.root_graph)
    if gotten_obj is not None:
        return value_ref.get_field().get_attribute(astc.nast.attr, from_module)

    if option is not None and option.eval_as_written_target:
        return attr
        
    # value is unknown
    if config.show_warnings:
        print('Assigning value is not found in L.{}'.format(astc.lineno))
    return None

def veval_ast_assign(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.Assign))
    lineprop = utils.LineProperty(astc.lineno)

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    value_obj = try_get_ref(value, 'assign', lineprop)

    if value is None:
        if config.show_warnings:
            print('It is possible that assiging value is invalid in L.{}'.format(astc.lineno))
        return None

    target_option = VEvalOption()
    target_option.eval_as_written_target = True
    targets = veval_ast(astc.c(astc.nast.targets[0]), local_field, graph, target_option)

    def return_value_or_ref(obj : 'value.Object'):
            if isinstance(obj.get_value(), values.NumberValue):
                return values.ValueRef(obj.get_value())

            if isinstance(obj.get_value(), values.StrValue):
                return values.ValueRef(obj.get_value())

            if isinstance(obj.get_value(), values.BoolValue):
                return values.ValueRef(obj.get_value())

            if isinstance(obj.get_value(), values.NoneValue):
                return values.ValueRef(obj.get_value())

            if isinstance(obj.get_value(), values.TupleValue):
                return values.ValueRef(obj.get_value())

            return obj

    if isinstance(targets, list):
        # ex. a,b = (1,2)
        if not isinstance(value_obj.get_value(), values.TupleValue):
            # TODO fix it
            assert(False)   # not supported
        
        for i in range(len(targets)):
            node_assign = nodes.NodeAssign(targets[i], value_obj.get_value().get_constant_value()[i], astc.lineno)
            targets[i].revise(try_get_ref(value_obj.get_value().get_constant_value()[i],'assign', lineprop))
            graph.add_node(node_assign)
    else:
        assigned_obj = return_value_or_ref(value_obj)
        node_assign = nodes.NodeAssign(targets, assigned_obj, astc.lineno)
        targets.revise(assigned_obj)
        graph.add_node(node_assign)

def veval_ast_name(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph', option : 'VEvalOption' = None) -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Name))

    from_module = True
    if option is not None and option.eval_as_written_target:
        from_module = False

    ret = local_field.get_attribute(astc.nast.id, from_module=from_module)
    return ret

def veval_ast_call(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Call))
    lineprop = utils.LineProperty(astc.lineno)

    func = veval_ast(astc.c(astc.nast.func), local_field, graph)
    if func == None or not func.has_obj():
        if config.show_warnings:
            print('Unknown function {} is called in L.{}'.format(get_ast_name_forcibly(astc.nast.func) ,astc.lineno))
        return None

    func_obj = try_get_ref(func, 'call', lineprop)
    func_value = try_get_value(func, 'call', lineprop)

    finput = functions.FunctionArgInput()

    for arg in astc.nast.args:
        arg_ = veval_ast(astc.c(arg), local_field, graph)
        finput.inputs.append(try_get_ref(arg_, 'call', lineprop))

    for keyword in astc.nast.keywords:
        arg_ = veval_ast(astc.c(keyword.value), local_field, graph)
        finput.keywords[keyword.arg] = try_get_ref(arg_, 'call', lineprop)

    lineprop = utils.LineProperty(astc.lineno)

    ret = None
    if isinstance(func_value, values.FuncValue):
        ret = func_value.func.vcall(local_field.module, graph, func_value.obj, finput, lineprop)
        return ret

    elif isinstance(func_value, values.Instance):
        # __call__
        call_func_ref = func_obj.try_get_and_store_obj('__call__', graph.root_graph)
        if call_func_ref is not None:        
            ret = call_func_ref.get_value().func.vcall(local_field.module, graph, func_obj, finput, lineprop)
            return ret

    
    if config.show_warnings:
        print('Unknown function is called in L.{}'.format(astc.lineno))
    return None

    
def veval_ast_return(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'None':
    assert(isinstance(astc.nast, gast.gast.Return))
    lineprop = utils.LineProperty(astc.lineno)

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    value_obj = try_get_ref(value, 'return', lineprop)
    value_value = try_get_value(value, 'return', lineprop)

    if value_value is None:
        if config.show_warnings:
            print('Returned values are not found. in L.{}'.format(astc.lineno))
        return None

    node = nodes.NodeReturn(value_value,astc.lineno)
    graph.add_node(node)
    return value_obj

def veval_ast_if(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.If))
    lineprop = utils.LineProperty(astc.lineno)

    # if condition
    test = veval_ast(astc.c(astc.nast.test), local_field, graph)
    test_value = try_get_value(test, 'if', lineprop)

    id_str = str(utils.get_guid())
    if_id = 'if_' + id_str
    true_id = 'true_' + id_str
    false_id = 'false_' + id_str

    # True
    values.push_history(true_id)

    true_graph = Graph()
    true_graph.root_graph = graph.root_graph
    true_graph.name = 'True'
    body = veval_ast(astc.c(astc.nast.body), local_field, true_graph)

    true_value_inputs = values.get_inputs()
    true_value_outputs = values.get_outputs()

    values.pop_history()

    # False
    values.push_history(false_id)

    false_graph = Graph()
    false_graph.root_graph = graph.root_graph
    false_graph.name = 'False'
    body = veval_ast(astc.c(astc.nast.orelse), local_field, false_graph)

    false_value_inputs = values.get_inputs()
    false_value_outputs = values.get_outputs()

    values.pop_history()

    # generate pairs
    value_pairs = {}
    for v in true_value_inputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['true_input_value'] = v.input_value
        value_pairs[key]['true_input_body_value'] = v.value

    for v in true_value_outputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['true_output_body_value'] = v.value

    for v in false_value_inputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['false_input_value'] = v.input_value
        value_pairs[key]['false_input_body_value'] = v.value

    for v in false_value_outputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['false_output_body_value'] = v.value

    inputs = []
    outputs = []

    for k, v in value_pairs.items():
        name = v['name']
        field = v['field']

        input_value = None
        true_input_body_value = None
        false_input_body_value = None
            
        if 'true_input_value' in v:
            input_value = v['true_input_value']
        elif 'false_input_value' in v:
            input_value = v['false_input_value']

        if input_value is not None:
            if 'true_input_body_value' in v:
                true_input_body_value = v['true_input_body_value']
            else:
                true_input_body_value = functions.generate_value_with_same_type(input_value)

            if 'false_input_body_value' in v:
                false_input_body_value = v['false_input_body_value']
            else:
                false_input_body_value = functions.generate_value_with_same_type(input_value)

        true_output_body_value = None
        false_output_body_value = None
        output_value = None

        if 'true_output_body_value' in v:
            true_output_body_value = v['true_output_body_value']
        else:
            true_output_body_value = true_input_body_value

        if 'false_output_body_value' in v:
            false_output_body_value = v['false_output_body_value']
        else:
            false_output_body_value = false_input_body_value

        # TODO check types between true and false

        if true_output_body_value is not None or false_output_body_value is not None:
            output_value = functions.generate_value_with_same_type(true_output_body_value)

        if input_value is not None:
            inputs.append(input_value)

            true_graph.add_input_value(true_input_body_value)
            false_graph.add_input_value(false_input_body_value)

        if output_value is not None:
            outputs.append(output_value)
            true_graph.add_output_value(true_output_body_value)
            false_graph.add_output_value(false_output_body_value)
            
            if field.get_attribute(name).has_obj():
                field.get_attribute(name).get_ref().revise(output_value)
            else:
                field.get_attribute(name).revise(values.ValueRef(output_value))

    node = nodes.NodeIf(test_value, inputs, true_graph, false_graph, astc.lineno)
    node.set_outputs(outputs)

    graph.add_node(node)

    return None

def veval_ast_aug_assign(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.AugAssign))
    lineprop = utils.LineProperty(astc.lineno)

    target = veval_ast(astc.c(astc.nast.target), local_field, graph)
    value = veval_ast(astc.c(astc.nast.value), local_field, graph)

    target_value = try_get_value(target, 'aug_assign', lineprop)
    value_value = try_get_value(value, 'aug_assign', lineprop)

    binop = nodes.BinOpType.Unknown
    if isinstance(astc.nast.op, gast.Add):
        binop = nodes.BinOpType.Add
    if isinstance(astc.nast.op, gast.Sub):
        binop = nodes.BinOpType.Sub

    node_aug_assign = nodes.NodeAugAssign(target_value, value_value, binop, astc.lineno)
    graph.add_node(node_aug_assign)

    # TODO : estimate type

    new_value = functions.generate_value_with_same_type(target_value)
    node_aug_assign.set_outputs([new_value])
    target.get_ref().revise(new_value)

def veval_ast_expr(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    call a function without not assigning
    Ex. b.x()
    '''
    assert(isinstance(astc.nast, gast.gast.Expr))
    return veval_ast(astc.c(astc.nast.value), local_field, graph)

def veval_ast_subscript(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. x[1], x[y,z]
    '''
    assert(isinstance(astc.nast, gast.gast.Subscript))
    lineprop = utils.LineProperty(astc.lineno)

    def veval_with_default(nast, default_value):
        if nast is None:
            ret = values.NumberValue(default_value)
            ret.name = '@SliceDefault'
            return ret
        obj = veval_ast(astc.c(nast), local_field, graph)
        return try_get_value(obj, 'subscript', lineprop)

    def get_slice_indices(slice):
        if slice.lower is None and slice.upper is None and slice.step is None:
            return []
        indices = [veval_with_default(slice.lower, 0),
                   veval_with_default(slice.upper, utils.slice_int_max)]
        if slice.step is not None:
            indices.append(veval_with_default(slice.step, 1))
        return indices

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    value_value = try_get_value(value, 'subscript', lineprop)

    if isinstance(astc.nast.slice, gast.gast.Index):
        slice_ = veval_ast(astc.c(astc.nast.slice.value), local_field, graph)
        slice_value = try_get_value(slice_, 'subscript', lineprop)

        if isinstance(slice_value, values.TupleValue):
            # ex. x[1,2]
            if slice_value.has_constant_value():
                values_ = [try_get_value(x, 'subscript', lineprop) for x in slice_value.get_constant_value()]
                node = nodes.NodeGetItem(value_value, values_, line=lineprop)
            else:
                if config.show_warnings:
                    print('This subscript is not supported. in L.{}'.format(astc.lineno))
                node = nodes.NodeInvalid(line=lineprop)
        else:
            # ex. x[1]
            node = nodes.NodeGetItem(value_value, [slice_value])
        ret_value = values.Value()
        node.set_outputs([ret_value])
        graph.add_node(node)
        return values.ValueRef(ret_value)

    elif isinstance(astc.nast.slice, gast.gast.Slice):

        indices = get_slice_indices(astc.nast.slice)

        node = nodes.NodeSlice(value_value, indices, [len(indices)])
        ret_value = functions.generate_value_with_same_type(value_value)
        node.set_outputs([ret_value])
        graph.add_node(node)
        return values.ValueRef(ret_value)

    elif isinstance(astc.nast.slice, gast.gast.ExtSlice):
        indices = []
        slice_specs = []
        for dim in astc.nast.slice.dims:
            if isinstance(dim, gast.gast.Index):
                indices.append(try_get_value(veval_ast(astc.c(dim.value), local_field, graph), 'subscript', lineprop))
                slice_specs.append(1)
            elif isinstance(dim, gast.gast.Slice):
                ni = get_slice_indices(dim)
                indices.extend(ni)
                slice_specs.append(len(ni))
            else:
                assert False, 'Unknown slice: %s in %s' % (dim, nast.slice)

        node = nodes.NodeSlice(value_value, indices, slice_specs)
        ret_value = functions.generate_value_with_same_type(value_value)
        node.set_outputs([ret_value])
        graph.add_node(node)
        return values.ValueRef(ret_value)

    return None

def veval_ast_listcomp(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. [x for x in xx]
    [elt for target in iter]
    '''
    assert(isinstance(astc.nast, gast.gast.ListComp))
    lineprop = utils.LineProperty(astc.lineno)

    listcomp_guid = str(utils.get_guid())
    listcomp_id = 'listcomp_' + listcomp_guid
    body_id = 'listcomp_body_' + listcomp_guid
    internal_counter_id = '@internal/listcomp_counter_' + listcomp_guid
    internal_list_id = '@internal/listcomp_list_' + listcomp_guid
    internal_cond_id = '@internal/listcomp_cond_' + listcomp_guid

    generator = astc.nast.generators[0]
    iter_value = try_get_value(veval_ast(astc.c(generator.iter), local_field, graph), 'generator', lineprop)
    list_value = values.ListValue()
    list_obj = values.ValueRef(list_value)

    node_generate_list = nodes.NodeGenerate('List', [], lineprop)
    node_generate_list.set_outputs([list_value])
    graph.add_node(node_generate_list)

    # body
    target_name = ''
    if isinstance(generator.target, gast.gast.Name):
        target_name = generator.target.id
    else:
        if config.show_warnings:
            print('This for is not supported. in L.{}'.format(astc.lineno))
        return None

    counter_value = values.NumberValue(None)
    counter_value.dtype = np.array(0).dtype
    counter_value.name = internal_counter_id

    cond_value = values.BoolValue(None)
    cond_value.name = internal_cond_id

    # set values with internal name
    local_field.get_attribute(internal_list_id).revise(list_obj)

    values.push_history(listcomp_id)

    body_graph = Graph()
    body_graph.root_graph = graph.root_graph
    body_graph.name = 'Body_' + listcomp_guid

    node_forgen = nodes.NodeForGenerator(counter_value, iter_value)

    target_ref = iter_value.get_iterator()
    if target_ref is None:
        target_ref = values.ValueRef(values.UnknownValue())
        if config.show_warnings:
            print('unknown iteratable type in L.{}'.format(astc.lineno))
    target_value = target_ref.get_value()

    node_forgen.set_outputs([target_ref.get_value()])
    local_field.get_attribute(target_name, from_module=False).revise(target_ref)
    
    body_graph.add_node(node_forgen)

    elt = veval_ast(astc.c(astc.nast.elt), local_field, body_graph)
    elt_obj = try_get_ref(elt, 'listcomp', lineprop)

    finput = functions.FunctionArgInput()
    finput.inputs.append(elt_obj)

    append_value = local_field.get_attribute(internal_list_id).get_ref().get_field().get_attribute('append').get_ref().get_value()
    append_value.func.vcall(local_field.module, body_graph, local_field.get_attribute(internal_list_id).get_ref(), finput, lineprop)

    value_inputs = values.get_inputs()
    value_outputs = values.get_outputs()

    values.pop_history()

    inputs = []
    outputs = []

    # default input for subgraph's input
    body_graph.add_input_value(counter_value)
    body_graph.add_input_value(cond_value)
    body_graph.add_input_value(iter_value)

    # default output for subgraph's output
    body_graph.add_output_value(cond_value)
    body_graph.add_output_value(iter_value)

    # default output
    outputs.append(functions.generate_value_with_same_type(iter_value))
    
    # generate pairs
    value_pairs = {}
    for v in value_inputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['input_value'] = v.input_value
        value_pairs[key]['input_body_value'] = v.value

    for v in value_outputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['output_body_value'] = v.value

    # remove iterator
    removed_name = str(local_field.id) + '_' + target_value.name
    del value_pairs[removed_name]

    for k, v in value_pairs.items():
        name = v['name']
        field = v['field']

        if 'input_body_value' in v:
            inputs.append(v['input_value'])
            body_graph.add_input_value(v['input_body_value'])

        else:
            temp_value1 = functions.generate_value_with_same_type(v['output_body_value'])
            temp_value2 = functions.generate_value_with_same_type(v['output_body_value'])
            inputs.append(temp_value1)
            body_graph.add_input_value(temp_value2)

        if 'output_body_value' in v:
            body_graph.add_output_value(v['output_body_value'])
            output_value = functions.generate_value_with_same_type(v['output_body_value'])
            outputs.append(output_value)
            if field.get_attribute(name).has_obj():
                field.get_attribute(name).get_ref().revise(output_value)
            else:
                field.get_attribute(name).revise(values.ValueRef(output_value))
        else:
            temp_value1 = v['input_body_value']
            temp_value2 = functions.generate_value_with_same_type(v['input_body_value'])
            body_graph.add_output_value(temp_value1)
            outputs.append(temp_value2)

    node = nodes.NodeListcomp(iter_value, inputs, body_graph, astc.lineno)
    node.set_outputs(outputs)

    graph.add_node(node)

    return local_field.get_attribute(internal_list_id).get_ref()

def veval_ast_bin_op(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    """
    eval binary operation.
    Ex. a + b, b // c, etc
    """
    assert(isinstance(astc.nast, gast.gast.BinOp))
    lineprop = utils.LineProperty(astc.lineno)

    left = veval_ast(astc.c(astc.nast.left), local_field, graph)
    right = veval_ast(astc.c(astc.nast.right), local_field, graph)

    left_value = try_get_value(left, 'compare', lineprop)
    right_value = try_get_value(right, 'compare', lineprop)

    binop = nodes.BinOpType.Unknown
    if isinstance(astc.nast.op, gast.Add):
        binop = nodes.BinOpType.Add
    if isinstance(astc.nast.op, gast.Sub):
        binop = nodes.BinOpType.Sub
    if isinstance(astc.nast.op, gast.Mult):
        binop = nodes.BinOpType.Mul

    node_bin_op = nodes.NodeBinOp(left_value, right_value, binop, astc.lineno)

    ret_value = veval_bin.veval(binop, left_value, right_value)

    node_bin_op.set_outputs([ret_value])
    graph.add_node(node_bin_op)

    return values.ValueRef(ret_value)

def veval_ast_unary_op(astc : 'AstContext', local_field : 'values.Field', graph : 'graphs.Graph'):
    """
    eval unary operation.
    Ex. -xx
    """
    assert(isinstance(astc.nast, gast.gast.UnaryOp))
    lineprop = utils.LineProperty(astc.lineno)

    unaryop = nodes.UnaryOpType.Unknown
    if isinstance(astc.nast.op, gast.UAdd):
        unaryop = nodes.UnaryOpType.UAdd
    if isinstance(astc.nast.op, gast.USub):
        unaryop = nodes.UnaryOpType.USub
    if isinstance(astc.nast.op, gast.Not):
        unaryop = nodes.UnaryOpType.Not

    operand = veval_ast(astc.c(astc.nast.operand), local_field, graph)
    operand_value = try_get_value(operand, 'unary', lineprop)

    node = nodes.NodeUnaryOp(operand_value, unaryop)

    ret_value = veval_unary.veval(unaryop, operand_value)

    node.set_outputs([ret_value])
    graph.add_node(node)

    return values.ValueRef(ret_value)


def veval_ast_compare(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    """
    eval Compare.
    Ex. a >= b, a != b, a is b, etc
    """
    assert(isinstance(astc.nast, gast.gast.Compare))
    lineprop = utils.LineProperty(astc.lineno)

    left = veval_ast(astc.c(astc.nast.left), local_field, graph)
    right = veval_ast(astc.c(astc.nast.comparators[0]), local_field, graph)

    left_value = try_get_value(left, 'compare', lineprop)
    right_value = try_get_value(right, 'compare', lineprop)

    compare = nodes.CompareType.unknown
    if isinstance(astc.nast.ops[0], gast.Eq):
        compare = nodes.CompareType.Eq
    if isinstance(astc.nast.ops[0], gast.NotEq):
        compare = nodes.CompareType.NotEq
    if isinstance(astc.nast.ops[0], gast.Is):
        compare = nodes.CompareType.Is
    if isinstance(astc.nast.ops[0], gast.IsNot):
        compare = nodes.CompareType.IsNot
    if isinstance(astc.nast.ops[0], gast.Gt):
        compare = nodes.CompareType.Gt
    if isinstance(astc.nast.ops[0], gast.GtE):
        compare = nodes.CompareType.GtE
    if isinstance(astc.nast.ops[0], gast.Lt):
        compare = nodes.CompareType.Lt
    if isinstance(astc.nast.ops[0], gast.LtE):
        compare = nodes.CompareType.LtE

    node_compare = nodes.NodeCompare(left_value, right_value, compare, astc.lineno)

    ret_value = values.BoolValue(None)
    ret_value.name = '@{}'.format(lineprop)
    node_compare.set_outputs([ret_value])
    graph.add_node(node_compare)

    return values.ValueRef(ret_value)


def veval_ast_num(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. 1, 2, ...
    '''
    assert(isinstance(astc.nast, gast.gast.Num))
    lineprop = utils.LineProperty(astc.lineno)
    value = values.NumberValue(astc.nast.n)
    ret = values.ValueRef(value)

    name = values.create_ref_value_name_with_constant(ret)
    ret.name = name
    ret.get_value().name = name
    return ret

def veval_ast_str(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. "str"
    '''
    assert(isinstance(astc.nast, gast.gast.Str))
    lineprop = utils.LineProperty(astc.lineno)
    value = values.StrValue(astc.nast.s)
    ret = values.ValueRef(value)

    name = values.create_ref_value_name_with_constant(ret)
    ret.name = name
    ret.get_value().name = name
    return ret

def veval_ast_name_constant(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. True
    '''
    assert(isinstance(astc.nast, gast.gast.NameConstant))
    lineprop = utils.LineProperty(astc.lineno)
    ret = None
    if astc.nast.value == True:
        ret = values.ValueRef(values.BoolValue(True))
    if astc.nast.value == False:
        ret = values.ValueRef(values.BoolValue(False))
    if astc.nast.value is None:
        ret = values.ValueRef(values.NoneValue())

    name = values.create_ref_value_name_with_constant(ret)
    ret.name = name
    ret.get_value().name = name
    return ret

def veval_ast_tuple(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph', option : 'VEvalOption' = None):
    assert(isinstance(astc.nast, gast.gast.Tuple))
    lineprop = utils.LineProperty(astc.lineno)

    if option is not None and option.eval_as_written_target:
        vs = []
        for v in astc.nast.elts:
            a_ = veval_ast(astc.c(v), local_field, graph, option=option)
            vs.append(a_)
        return vs
    else:
        vs_ref = []
        vs = []

        for v in astc.nast.elts:
            a_ = veval_ast(astc.c(v), local_field, graph, option=option)
            v_ = try_get_ref(a_, 'tuple', lineprop)
            vs_ref.append(v_)
            vs.append(v_.get_value())
            v_.in_container = True

        tuple_value = values.TupleValue(vs_ref)

        node = nodes.NodeGenerate('Tuple', vs, line=lineprop)
        node.set_outputs([tuple_value])
        graph.add_node(node)
        
        return values.ValueRef(tuple_value)

def veval_ast_list(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.List))
    '''
    Ex. [],[x,y,z]
    TODO : Initializer
    '''
    lineprop = utils.LineProperty(astc.lineno)

    elts = []
    for elt in astc.nast.elts:
        elt_ = veval_ast(astc.c(elt), local_field, graph)
        elt_obj = try_get_ref(elt_,'list', lineprop)
        elts.append(elt_obj)

    node = nodes.NodeGenerate('List', [elt.get_value() for elt in elts], lineprop)
    graph.add_node(node)
    value = values.ListValue(elts)
    node.set_outputs([value])

    return values.ValueRef(value)

def veval_ast_for_unroll(astc : 'AstContext', target_name, iter_ : 'values.ListValue', local_field : 'values.Field', graph : 'Graph'):
    '''
    for target in iter: 
        ...
    with unroll
    '''
    assert(isinstance(astc.nast, gast.gast.For))
    lineprop = utils.LineProperty(astc.lineno)

    for element in iter_.get_constant_value():
        local_field.get_attribute(target_name).revise(element)
        veval_ast(astc.c(astc.nast.body), local_field, graph)
    
    return None

def veval_ast_for(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    for target in iter:
        ...
    '''
    assert(isinstance(astc.nast, gast.gast.For))
    lineprop = utils.LineProperty(astc.lineno)

    # for target in iter:
    iter_ = veval_ast(astc.c(astc.nast.iter), local_field, graph)
    input_iter_value = try_get_value(iter_, 'for', lineprop)
    body_iter_value = functions.generate_value_with_same_type(input_iter_value, suffix_type=functions.SuffixType.Input)

    # get target name
    target_name = ''
    if isinstance(astc.nast.target, gast.gast.Name):
        target_name = astc.nast.target.id
    else:
        if config.show_warnings:
            print('This for is not supported. in L.{}'.format(astc.lineno))
        return None

    # unroll?
    if isinstance(input_iter_value, values.ListValue) and input_iter_value.has_constant_value() and input_iter_value.dtype is None:
        return veval_ast_for_unroll(astc, target_name, input_iter_value, local_field, graph)

    for_guid = utils.get_guid()
    for_id = 'for_' + str(for_guid)
    body_id = 'body_' + str(for_guid)

    values.push_history(for_id)

    # body
    body_graph = Graph()
    body_graph.root_graph = graph.root_graph
    body_graph.name = 'Body_' + str(for_guid)

    # generate a node for input
    node_input = nodes.NodeInput('input')
    body_graph.add_node(node_input)

    body_counter_value = values.NumberValue(None)
    body_counter_value.dtype = np.array(0).dtype
    body_counter_value.name = 'for_counter_' + str(for_guid)

    body_cond_value = values.BoolValue(None)
    body_cond_value.name = 'for_cond_' + str(for_guid)
    
    # create a node to lookup a value from sequence
    node_forgen = nodes.NodeForGenerator(body_counter_value, body_iter_value)

    # generate iterator
    target_ref = input_iter_value.get_iterator()
    if target_ref is None:
        target_ref = values.ValueRef(values.UnknownValue())
        if config.show_warnings:
            print('unknown iteratable type in L.{}'.format(astc.lineno))
    target_value = target_ref.get_value()
    
    node_forgen.set_outputs([target_ref.get_value()])

    target_attribute = local_field.get_attribute(target_name)
    target_attribute.revise(target_ref)
    body_graph.add_node(node_forgen)

    # veval body
    body = veval_ast(astc.c(astc.nast.body), local_field, body_graph)

    value_inputs = values.get_inputs()
    value_outputs = values.get_outputs()

    values.pop_history()

    inputs = []
    outputs = []
    node_input_outputs = []

    # default input for subgraph's input
    body_graph.add_input_value(body_counter_value)
    body_graph.add_input_value(body_cond_value)
    body_graph.add_input_value(body_iter_value)

    # default output for subgraph's output
    body_graph.add_output_value(body_cond_value)
    body_graph.add_output_value(body_iter_value)

    # default output
    outputs.append(functions.generate_value_with_same_type(input_iter_value))
    
    # generate pairs
    value_pairs = {}
    for v in value_inputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['input_value'] = v.input_value
        value_pairs[key]['input_body_value'] = v.value

    for v in value_outputs:
        key = str(v.field.id) + '_' + v.name
        if not (key in value_pairs.keys()):
            value_pairs[key] = {}

        value_pairs[key]['field'] = v.field
        value_pairs[key]['name'] = v.name
        value_pairs[key]['output_body_value'] = v.value

    for k, v in value_pairs.items():
        name = v['name']
        field = v['field']

        if 'input_body_value' in v:
            inputs.append(v['input_value'])

            body_graph.add_input_value(v['input_body_value'])
        else:
            temp_value1 = functions.generate_value_with_same_type(v['output_body_value'], is_dummy_value=True, suffix_type=functions.SuffixType.Dummy)
            temp_value2 = functions.generate_value_with_same_type(v['output_body_value'], suffix_type=functions.SuffixType.Dummy)
            inputs.append(temp_value1)

            body_graph.add_input_value(temp_value2)
            node_input_outputs.append(temp_value2)

        if 'output_body_value' in v:
            body_graph.add_output_value(v['output_body_value'])
            output_value = functions.generate_value_with_same_type(v['output_body_value'])
            outputs.append(output_value)
            if field.get_attribute(name).has_obj():
                field.get_attribute(name).get_ref().revise(output_value)
            else:
                field.get_attribute(name).revise(values.ValueRef(output_value))
        else:
            temp_value1 = v['input_body_value']
            temp_value2 = functions.generate_value_with_same_type(v['input_body_value'])
            body_graph.add_output_value(temp_value1)
            outputs.append(temp_value2)

    node = nodes.NodeFor(input_iter_value, inputs, body_graph, astc.lineno)
    node.set_outputs(outputs)
    node_input.set_outputs(node_input_outputs)

    graph.add_node(node)

    return None

def veval_ast(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph', option : 'VEvalOption' = None):
    if isinstance(astc.nast, list):
        ret = None
        for nast_ in astc.nast:
            ret = veval_ast(AstContext(nast_, astc.lineno_offset), local_field, graph)
            if ret is not None:
                break
        return ret

    elif isinstance(astc.nast, gast.gast.Assign):
        veval_ast_assign(astc, local_field, graph)
        return None

    elif isinstance(astc.nast, gast.gast.Attribute):
        ret = veval_ast_attribute(astc, local_field, graph, option)
        return ret

    elif isinstance(astc.nast, gast.gast.Call):
        ret = veval_ast_call(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.BinOp):
        ret = veval_ast_bin_op(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.UnaryOp):
        ret = veval_ast_unary_op(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.Compare):
        ret = veval_ast_compare(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.Return):
        ret = veval_ast_return(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.Name):
        ret = veval_ast_name(astc, local_field, graph, option)
        return ret

    elif isinstance(astc.nast, gast.gast.AugAssign):
        veval_ast_aug_assign(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.Expr):
        veval_ast_expr(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.Subscript):
        return veval_ast_subscript(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.ListComp):
        return veval_ast_listcomp(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.If):
        veval_ast_if(astc, local_field, graph)
        return None

    elif isinstance(astc.nast, gast.gast.Num):
        ret = veval_ast_num(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.Str):
        ret = veval_ast_str(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.NameConstant):
        ret = veval_ast_name_constant(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.Tuple):
        ret = veval_ast_tuple(astc, local_field, graph, option)
        return ret

    elif isinstance(astc.nast, gast.gast.List):
        ret = veval_ast_list(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.For):
        veval_ast_for(astc, local_field, graph)
        return None
    else:
        if config.show_warnings:
            print('Unknown ast is found : {} in L.{}'.format(type(astc.nast),astc.lineno))
