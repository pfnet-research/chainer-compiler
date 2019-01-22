import chainer
import chainer.functions as F
import chainer.links as L
import inspect
import ast, gast
import weakref

from elichika.parser import config
from elichika.parser import nodes
from elichika.parser import values
from elichika.parser import functions
from elichika.parser import utils
from elichika.parser.graphs import Graph
from elichika.parser import veval_bin
from elichika.parser import veval_unary

def try_get_value(value, name, lineprop, is_none_allowed = False):
    if value is None:
        if config.show_warnings:
            print('Failed to get value {}. in L.{}'.format(name, lineprop))
        return None
    
    if isinstance(value, values.NoneValue) and not is_none_allowed:
        if config.show_warnings:
            print('Value {} is none. in L.{}'.format(name, lineprop))
        return None

    if isinstance(value, values.Value):
        return value

    if isinstance(value, values.Attribute):
        return value.get_value()

    raise Exception('Value {} is invalid. in L.{}'.format(name, lineprop))

def get_ast_name_forcibly(ast):
    if isinstance(ast, gast.gast.Name):
        return ast.id
    if isinstance(ast, gast.gast.Attribute):
        return ast.attr
    return ''

def get_input_attritubtes(target : 'values.Field', commit_id1 : 'str', commit_id2 : 'str'):
    ret = [v for v in target.attributes.values() if v.has_accessed(commit_id1, commit_id2) and v.get_value(False).get_field() is None]
    dicts = [v.get_value(False) for v in target.attributes.values() if v.get_value(False).get_field() is not None]

    for d in dicts:
        ret_ = get_input_attritubtes(d.get_field(), commit_id1, commit_id2)
        ret.extend(ret_)

    return ret

def get_output_attritubtes(target : 'values.Field', commit_id1 : 'str', commit_id2 : 'str'):
    ret = [v for v in target.attributes.values() if v.has_diff(commit_id1, commit_id2) or v.get_value(False).has_diff(commit_id1, commit_id2)]
    dicts = [v.get_value(False) for v in target.attributes.values() if v.get_value(False).get_field() is not None]

    for d in dicts:
        ret_ = get_output_attritubtes(d.get_field(), commit_id1, commit_id2)
        ret.extend(ret_)

    return ret

def filter_attributes(attributes):
    ret = []

    for attribute in attributes:
        if attribute.has_value() and isinstance(attribute.get_value(False), values.Instance):
            continue

        if attribute.has_value() and isinstance(attribute.get_value(False), values.FuncValue):
            continue

        ret.append(attribute)

    return ret

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

def veval_ast_attribute(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Attribute))

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    attr = value.get_value().get_field().get_attribute(astc.nast.attr)

    if attr.has_value():
        return attr

    # if attr is not found
    gotten_value = value.get_value().try_get_and_store_value(astc.nast.attr)
    if gotten_value is not None:
        return attr

    return attr

def veval_ast_assign(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.Assign))

    targets = veval_ast(astc.c(astc.nast.targets[0]), local_field, graph)

    isTuple = False
    if isinstance(targets, values.TupleValue):
        targets = targets.values 
        isTuple = True

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)

    if value is None:
        if config.show_warnings:
            print('It is possible that assiging value is invalid in L.{}'.format(astc.lineno))    
        return None

    if isTuple:
        for i in range(len(targets)):
            node_assign = nodes.NodeAssign(targets[i], value.values[i].get_value(), astc.lineno)
            targets[i].revise(value.values[i].get_value())
            graph.add_node(node_assign)
    else:
        node_assign = nodes.NodeAssign(targets, value, astc.lineno)
        targets.revise(value.get_value())
        graph.add_node(node_assign)

def veval_ast_name(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Name))

    ret = local_field.get_attribute(astc.nast.id)
    return ret

def veval_ast_call(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Call))

    func = veval_ast(astc.c(astc.nast.func), local_field, graph)
    if func == None or not func.has_value():
        if config.show_warnings:
            print('Unknown function {} is called in L.{}'.format(get_ast_name_forcibly(astc.nast.func) ,astc.lineno)) 
        return None

    func_value = func.get_value()

    args = []
    fargs = []
    for arg in astc.nast.args:
        arg_ = veval_ast(astc.c(arg), local_field, graph)
        farg = functions.FunctionArg()
        farg.value = arg_.get_value() 
        args.append(farg.value)
        fargs.append(farg)

    for keyword in astc.nast.keywords:
        arg_ = veval_ast(astc.c(keyword.value), local_field, graph)
        farg = functions.FunctionArg()
        farg.name = keyword.arg
        farg.value = arg_.get_value() 
        args.append(farg.value)
        fargs.append(farg)

    lineprop = utils.LineProperty(astc.lineno)

    ret = None
    if isinstance(func_value, values.FuncValue):
        ret = func_value.func.vcall(local_field.module, graph, func_value.value, fargs, lineprop)

    elif isinstance(func_value, values.Instance) and func_value.callable:
        # __call__
        ret = func_value.func.func.vcall(local_field.module, graph, func_value, fargs, lineprop)

    else:
        if config.show_warnings:
            print('Unknown function is called in L.{}'.format(astc.lineno)) 
        return None

    return ret

def veval_ast_return(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'None':
    assert(isinstance(astc.nast, gast.gast.Return))

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)

    if not value.has_value():
        if config.show_warnings:
            print('Returned values are not found. in L.{}'.format(astc.lineno))    
        return None

    node = nodes.NodeReturn(value.get_value(),astc.lineno)
    graph.add_node(node)
    return value.get_value()

def veval_ast_if(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.If))

    test = veval_ast(astc.c(astc.nast.test), local_field, graph)

    if_id = 'if_' + str(utils.get_guid())
    true_id = 'true_' + str(utils.get_guid())
    false_id = 'false_' + str(utils.get_guid())

    values.commit(if_id)

    # True condition
    values.checkout(if_id)
    true_graph = Graph()
    true_graph.name = 'True'
    body = veval_ast(astc.c(astc.nast.body), local_field, true_graph)

    values.commit(true_id)
    true_input_attributes = get_input_attritubtes(local_field, if_id, true_id)
    true_output_attributes = get_output_attritubtes(local_field, if_id, true_id)

    #TODO(durswd): improve
    true_input_attributes = filter_attributes(true_input_attributes)
    true_output_attributes = filter_attributes(true_output_attributes)

    true_output_attributes_2_values = {}

    for attribute in true_output_attributes:
        true_output_attributes_2_values[attribute] = attribute.get_value()

    # False condition
    values.checkout(if_id)
    false_graph = Graph()
    false_graph.name = 'False'
    orelse = veval_ast(astc.c(astc.nast.orelse), local_field, false_graph)

    values.commit(false_id)
    false_input_attributes = get_input_attritubtes(local_field, if_id, false_id)
    false_output_attributes = get_output_attritubtes(local_field, if_id, false_id)

    #TODO(durswd): improve
    false_input_attributes = filter_attributes(false_input_attributes)
    false_output_attributes = filter_attributes(false_output_attributes)

    false_output_attributes_2_values = {}

    for attribute in false_output_attributes:
        false_output_attributes_2_values[attribute] = attribute.get_value()

    # Merge
    values.checkout(if_id)

    # Input
    input_attributes = set(true_input_attributes) | set(false_input_attributes)

    # remove unexisting values
    input_attributes = [v for v in input_attributes if v.has_value()]
    input_values = [i.get_value() for i in input_attributes]

    # Output
    name2output_attributes = {}

    # generate attribute pairs
    for attribute in true_output_attributes:
        key = str(attribute.parent.id) + '_' + attribute.name

        if key in name2output_attributes.keys():
            name2output_attributes[key][0] = attribute
        else:
            name2output_attributes[key] = [attribute, None]

    for attribute in false_output_attributes:
        key = str(attribute.parent.id) + '_' + attribute.name

        if key in name2output_attributes.keys():
            name2output_attributes[key][1] = attribute
        else:
            name2output_attributes[key] = [None, attribute]

    output_attributes = set(true_output_attributes) | set(false_output_attributes)
    output_values = []

    non_volatiles = []

    for attribute_pair in name2output_attributes.values():
        true_attribute, false_attribute = attribute_pair
        name = ''
        parent = None # type: values.Field

        if true_attribute is not None:
            name = true_attribute.name
            parent = true_attribute.parent

        if false_attribute is not None:
            name = false_attribute.name
            parent = false_attribute.parent

        if true_attribute is not None:
            true_value = true_output_attributes_2_values[true_attribute]
        else:
            if parent.has_attribute(name):
                true_value = parent.get_attribute(name).get_value()

                if not true_value in input_values:
                    input_values.append(true_value)
            else:
                # TODO : it should be better
                # case
                # if xxx:
                #     y = 10
                # print(y)
                true_value = false_output_attributes_2_values[false_attribute]

        if false_attribute is not None:
            false_value = false_output_attributes_2_values[false_attribute]
        else:
            if parent.has_attribute(name):
                false_value = parent.get_attribute(name).get_value()

                if not false_value in input_values:
                    input_values.append(false_value)

            else:
                # TODO : it should be better
                # case
                # if xxx:
                #     y = 10
                # print(y)
                false_value = true_output_attributes_2_values[true_attribute]

        true_graph.add_output_value(true_value)
        false_graph.add_output_value(false_value)

        if true_attribute is not None and false_attribute is not None and true_attribute != false_attribute:
            # dynamic
            value = functions.generate_value_with_same_type(true_value)
            output_values.append(value)
            parent.get_attribute(name).revise(value)

        elif true_attribute is not None and false_attribute is not None:
            # change both
            value = functions.generate_value_with_same_type(true_value)
            output_values.append(value)

            if parent.get_attribute(name).is_non_volatile:
                non_volatiles.append((parent.get_attribute(name).initial_value, value))

            parent.get_attribute(name).revise(value)

        elif true_attribute in input_attributes:
            value = functions.generate_value_with_same_type(true_value)
            
            if parent.get_attribute(name).is_non_volatile:
                non_volatiles.append((parent.get_attribute(name).initial_value, value))

            output_values.append(value)
            parent.get_attribute(name).revise(value)
        else:
            value = functions.generate_value_with_same_type(false_value)
            
            if parent.get_attribute(name).is_non_volatile:
                non_volatiles.append((parent.get_attribute(name).initial_value, value))

            output_values.append(value)
            parent.get_attribute(name).revise(value)

    for input_value in input_values:
        true_graph.add_input_value(input_value)
        false_graph.add_input_value(input_value)

    node = nodes.NodeIf(test.get_value(), input_values, true_graph, false_graph, astc.lineno)
    node.set_outputs(output_values)

    graph.add_node(node)

    # add non-volatiles
    for tv, v in non_volatiles:
        node_nv = nodes.NodeNonVolatileAssign(tv, v)
        graph.add_node(node_nv)

    return None

def veval_ast_aug_assign(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.AugAssign))

    target = veval_ast(astc.c(astc.nast.target), local_field, graph)
    value = veval_ast(astc.c(astc.nast.value), local_field, graph)

    binop = nodes.BinOpType.Unknown
    if isinstance(astc.nast.op, gast.Add):
        binop = nodes.BinOpType.Add
    if isinstance(astc.nast.op, gast.Sub):
        binop = nodes.BinOpType.Sub

    target_value = target.get_value()
    if isinstance(target_value, values.NumberValue):
        node_aug_assign = nodes.NodeValueAugAssign(target_value, value.get_value(), binop, astc.lineno)
        new_value = functions.generate_value_with_same_type(target_value)
        target.revise(new_value)
        node_aug_assign.set_outputs([new_value])
        graph.add_node(node_aug_assign)
        return values.NoneValue()

    else:
        node_aug_assign = nodes.NodeAugAssign(target_value, value.get_value(), binop, astc.lineno)
        target_value.modify(node_aug_assign, None)   
        graph.add_node(node_aug_assign)
        return values.NoneValue()

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

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)

    if isinstance(astc.nast.slice, gast.gast.Index):
        slice_value = veval_ast(astc.c(astc.nast.slice.value), local_field, graph)
        node = nodes.NodeGetItem(value.get_value(), slice_value.get_value())
        ret_value = values.Value()
        node.set_outputs([ret_value])
        graph.add_node(node)
        return ret_value

    elif isinstance(astc.nast.slice, gast.gast.Slice):
        lower = veval_ast(astc.c(astc.nast.slice.lower), local_field, graph)
        upper = veval_ast(astc.c(astc.nast.slice.upper), local_field, graph)
        node = nodes.NodeSlice(value.get_value(), lower.get_value(), upper.get_value())
        ret_value = values.Value()
        node.set_outputs([ret_value])
        graph.add_node(node)
        return ret_value

    return None

def veval_ast_listcomp(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. [x for x in xx]
    [elt for target in iter]
    '''
    assert(isinstance(astc.nast, gast.gast.ListComp))
    lineprop = utils.LineProperty(astc.lineno)

    listcomp_id = 'listcomp_' + str(utils.get_guid())
    body_id = 'listcomp_body_' + str(utils.get_guid())
    
    values.commit(listcomp_id)

    generator = astc.nast.generators[0]
    iter_value = try_get_value(veval_ast(astc.c(generator.iter), local_field, graph), 'generator', lineprop)


    lst = values.ListValue()

    # body
    target_name = ''
    if isinstance(generator.target, gast.gast.Name):
        target_name = generator.target.id
    else:
        if config.show_warnings:
            print('This for is not supported. in L.{}'.format(astc.lineno))    
        return None

    target = values.Value()
    body_field = values.Field(local_field.module, local_field)
    body_field.get_attribute(target_name).revise(target)
    body_graph = Graph()
    body_graph.name = 'Body'

    elt = veval_ast(astc.c(astc.nast.elt), body_field, body_graph)
    farg = functions.FunctionArg()
    farg.name = ''
    farg.value = elt
    lst.append_func.func.vcall(local_field.module, body_graph, lst, [farg], lineprop)
        
    values.commit(listcomp_id)

    body_output_attributes = get_input_attritubtes(body_field, listcomp_id, body_id)
    body_intput_attributes = get_output_attritubtes(body_field, listcomp_id, body_id)

    for attribute in body_output_attributes:
        ifoutput_node = nodes.NodeForOutput(attribute.get_value(), astc.lineno)
        body_graph.add_node(ifoutput_node)

    # Exports
    values.checkout(listcomp_id)

    input_attributes = set(body_intput_attributes)
    inputs = [i.get_value() for i in input_attributes]

    output_attributes = set(body_output_attributes)
    outputs = []

    for attribute in output_attributes:
        value = values.Value()
        outputs.append(value)
        attribute.revise(value)

    node = nodes.NodeListcomp(iter_value, inputs, body_graph, astc.lineno)
    node.set_outputs(outputs)
    
    graph.add_node(node)


    # compare

    return lst

def veval_ast_bin_op(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    """
    eval binary operation.
    Ex. a + b, b // c, etc
    """
    assert(isinstance(astc.nast, gast.gast.BinOp))

    left = veval_ast(astc.c(astc.nast.left), local_field, graph)
    right = veval_ast(astc.c(astc.nast.right), local_field, graph)

    binop = nodes.BinOpType.Unknown
    if isinstance(astc.nast.op, gast.Add):
        binop = nodes.BinOpType.Add
    if isinstance(astc.nast.op, gast.Sub):
        binop = nodes.BinOpType.Sub

    node_bin_op = nodes.NodeBinOp(left.get_value(), right.get_value(), binop, astc.lineno)

    ret_value = veval_bin.veval(binop, left.get_value(), right.get_value())

    node_bin_op.set_outputs([ret_value])
    graph.add_node(node_bin_op)

    return ret_value

def veval_ast_unary_op(astc : 'AstContext', local_field : 'values.Field', graph : 'graphs.Graph'):
    """
    eval unary operation.
    Ex. -xx
    """
    assert(isinstance(astc.nast, gast.gast.UnaryOp))

    unaryop = nodes.UnaryOpType.Unknown
    if isinstance(astc.nast.op, gast.UAdd):
        unaryop = nodes.UnaryOpType.UAdd
    if isinstance(astc.nast.op, gast.USub):
        unaryop = nodes.UnaryOpType.USub
    if isinstance(astc.nast.op, gast.Not):
        unaryop = nodes.UnaryOpType.Not

    operand = veval_ast(astc.c(astc.nast.operand), local_field, graph)

    node = nodes.NodeUnaryOp(operand.get_value(), unaryop)

    ret_value = veval_unary.veval(unaryop, operand.get_value())

    node.set_outputs([ret_value])
    graph.add_node(node)

    return ret_value


def veval_ast_compare(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    """
    eval Compare.
    Ex. a >= b, a != b, a is b, etc
    """
    assert(isinstance(astc.nast, gast.gast.Compare))

    left = veval_ast(astc.c(astc.nast.left), local_field, graph)
    right = veval_ast(astc.c(astc.nast.comparators[0]), local_field, graph)

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

    node_compare = nodes.NodeCompare(left.get_value(), right.get_value(), compare, astc.lineno)

    ret_value = values.BoolValue(None)
    node_compare.set_outputs([ret_value])
    graph.add_node(node_compare)

    return ret_value


def veval_ast_num(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. 1, 2, ...
    '''
    assert(isinstance(astc.nast, gast.gast.Num))
    lineprop = utils.LineProperty(astc.lineno)
    value = values.NumberValue(astc.nast.n)
    return value

def veval_ast_name_constant(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. True
    '''
    assert(isinstance(astc.nast, gast.gast.NameConstant))
    lineprop = utils.LineProperty(astc.lineno)
    if astc.nast.value == True:
        return values.BoolValue(True)
    if astc.nast.value == False:
        return values.BoolValue(False)
    if astc.nast.value is None:
        return values.BoolValue(False)
    return None

def veval_ast_tuple(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.Tuple))
    vs = []
    for v in astc.nast.elts:
        v_ = veval_ast(astc.c(v), local_field, graph)
        vs.append(v_)

    return values.TupleValue(vs)

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
        elt_value = elt_.get_value()
        elts.append(elt_value)

    node = nodes.NodeGenerate('List', elts, lineprop)
    graph.add_node(node)
    value = values.ListValue()
    value.values.extend(elts)

    node.set_outputs([value])
    return value

def veval_ast_for(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    for target in iter:
        ...
    '''
    assert(isinstance(astc.nast, gast.gast.For))

    # for target in iter:
    iter_ = veval_ast(astc.c(astc.nast.iter), local_field, graph)

    # target
    target_name = ''
    if isinstance(astc.nast.target, gast.gast.Name):
        target_name = astc.nast.target.id
    else:
        if config.show_warnings:
            print('This for is not supported. in L.{}'.format(astc.lineno))    
        return None

    for_id = 'for_' + str(utils.get_guid())
    body_id = 'body_' + str(utils.get_guid())
    values.commit(for_id)

    # Body
    target = values.Value()
    local_field.get_attribute(target_name).revise(target)
    body_graph = Graph()
    body_graph.name = 'Body'
    body = veval_ast(astc.c(astc.nast.body), local_field, body_graph)

    values.commit(body_id)

    body_output_attributes = get_input_attritubtes(local_field, for_id, body_id)
    body_intput_attributes = get_output_attritubtes(local_field, for_id, body_id)

    # Exports
    values.checkout(for_id)

    input_attributes = set(body_intput_attributes)
    inputs = [i.get_value() for i in input_attributes]

    output_attributes = set(body_output_attributes)
    outputs = []

    non_volatiles = []

    for attribute in output_attributes:
        # FIXME
        value = values.Value()
        outputs.append(value)

        if attribute.is_non_volatile:
            non_volatiles.append((attribute.initial_value,value))
        attribute.revise(value)

    node = nodes.NodeFor(iter_, inputs, body_graph, astc.lineno)
    node.set_outputs(outputs)
    
    graph.add_node(node)

    # add non-volatiles
    for tv, v in non_volatiles:
        node_nv = nodes.NodeNonVolatileAssign(tv, v)
        graph.add_node(node_nv)

    return None

def veval_ast(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
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
        ret = veval_ast_attribute(astc, local_field, graph)
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
        ret = veval_ast_name(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.AugAssign):
        veval_ast_aug_assign(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.Expr):
        veval_ast_expr(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.Subscript):
        return veval_ast_subscript(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.ListComp):
        veval_ast_listcomp(astc, local_field, graph)

    elif isinstance(astc.nast, gast.gast.If):
        veval_ast_if(astc, local_field, graph)
        return None

    elif isinstance(astc.nast, gast.gast.Num):
        ret = veval_ast_num(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.NameConstant):
        ret = veval_ast_name_constant(astc, local_field, graph)
        return ret

    elif isinstance(astc.nast, gast.gast.Tuple):
        ret = veval_ast_tuple(astc, local_field, graph)
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