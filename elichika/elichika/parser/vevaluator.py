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


def get_attritubtes_with_diff(target, commit_id1 : 'str', commit_id2 : 'str'):
    ret = [v for v in target if v.has_diff(commit_id1, commit_id2)]
    dicts = [v.get_value() for v in target if v.get_value().get_field() is not None]

    for d in dicts:
        ret_ = get_attritubtes_with_diff(d.get_field().attributes.values(), commit_id1, commit_id2)
        ret.extend(ret_)

    return ret

def get_attritubtes_with_accessed(target, commit_id1 : 'str', commit_id2 : 'str'):
    ret = [v for v in target if v.has_accessed(commit_id1, commit_id2) and v.get_value().get_field() is None]
    dicts = [v.get_value() for v in target if v.get_value().get_field() is not None]

    for d in dicts:
        ret_ = get_attritubtes_with_accessed(d.get_field().attributes.values(), commit_id1, commit_id2)
        ret.extend(ret_)

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

    # if attr is func and not assigned
    func = value.get_value().try_get_func(astc.nast.attr)
    if func is not None:
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
            print('Unknown function is called in L.{}'.format(astc.lineno)) 
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
    true_field = values.Field(local_field.module, local_field)
    true_graph = Graph()
    true_graph.name = 'True'
    body = veval_ast(astc.c(astc.nast.body), true_field, true_graph)

    values.commit(true_id)
    true_output_attributes = get_attritubtes_with_diff(true_field.attributes_from_parent, if_id, true_id)
    true_intput_attributes = get_attritubtes_with_accessed(true_field.attributes_from_parent, if_id, true_id)

    for attribute in true_output_attributes:
        ifoutput_node = nodes.NodeIfOutput(attribute.get_value(), astc.lineno)
        true_graph.add_node(ifoutput_node)

    # False condition
    values.checkout(if_id)
    false_field = values.Field(local_field.module, local_field)
    false_graph = Graph()
    false_graph.name = 'False'
    orelse = veval_ast(astc.c(astc.nast.orelse), false_field, false_graph)

    values.commit(false_id)
    false_output_attributes = get_attritubtes_with_diff(false_field.attributes_from_parent, if_id, false_id)
    false_intput_attributes = get_attritubtes_with_accessed(false_field.attributes_from_parent, if_id, false_id)

    for attribute in false_output_attributes:
        ifoutput_node = nodes.NodeIfOutput(attribute.get_value(), astc.lineno)
        false_graph.add_node(ifoutput_node)

    # Merge
    values.checkout(if_id)

    input_attributes = set(true_output_attributes) | set(false_output_attributes)
    inputs = [i.get_value() for i in input_attributes]

    output_attributes = set(true_output_attributes) | set(false_output_attributes)
    outputs = []

    for attribute in output_attributes:
        value = values.Value()
        outputs.append(value)
        attribute.revise(value)

    node = nodes.NodeIf(test.get_value(), inputs, true_graph, false_graph, astc.lineno)

    node.set_outputs(outputs)

    graph.add_node(node)

    return None

def veval_ast_aug_assign(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.AugAssign))

    target = veval_ast(astc.c(astc.nast.target), local_field, graph)
    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    node_aug_assign = nodes.NodeAugAssign(target.get_value(), value.get_value(), astc.lineno)

    ret_value = values.Value()
    node_aug_assign.set_outputs([ret_value])
   
    target.revise(ret_value)
    graph.add_node(node_aug_assign)

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

    ret_value = functions.generateValueWithSameType(left.get_value())

    # TODO fixme
    if ret_value is None:
        ret_value = values.Value()

    node_bin_op.set_outputs([ret_value])
    graph.add_node(node_bin_op)

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
    assert(isinstance(astc.nast, gast.gast.Num))
    return values.NumberValue(astc.nast.n)

def veval_ast_tuple(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.Tuple))
    vs = []
    for v in astc.nast.elts:
        v_ = veval_ast(astc.c(v), local_field, graph)
        vs.append(v_)

    return values.TupleValue(vs)

def veval_ast_for(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
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
    body_field = values.Field(local_field.module, local_field)
    body_field.get_attribute(target_name).revise(target)
    body_graph = Graph()
    body_graph.name = 'Body'
    body = veval_ast(astc.c(astc.nast.body), body_field, body_graph)

    values.commit(body_id)

    body_output_attributes = get_attritubtes_with_diff(body_field.attributes_from_parent, for_id, body_id)
    body_intput_attributes = get_attritubtes_with_accessed(body_field.attributes_from_parent, for_id, body_id)

    for attribute in body_output_attributes:
        ifoutput_node = nodes.NodeForOutput(attribute.get_value(), astc.lineno)
        body_graph.add_node(ifoutput_node)

    # Exports
    values.checkout(for_id)

    input_attributes = set(body_intput_attributes)
    inputs = [i.get_value() for i in input_attributes]

    output_attributes = set(body_output_attributes)
    outputs = []

    for attribute in output_attributes:
        value = values.Value()
        outputs.append(value)
        attribute.revise(value)

    node = nodes.NodeFor(iter_, inputs, body_graph, astc.lineno)
    node.set_outputs(outputs)
    
    graph.add_node(node)

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

    elif isinstance(astc.nast, gast.gast.If):
        veval_ast_if(astc, local_field, graph)
        return None
    elif isinstance(astc.nast, gast.gast.Num):
        ret = veval_ast_num(astc, local_field, graph)
        return ret
    elif isinstance(astc.nast, gast.gast.Tuple):
        ret = veval_ast_tuple(astc, local_field, graph)
        return ret
    elif isinstance(astc.nast, gast.gast.For):
        veval_ast_for(astc, local_field, graph)
        return None
    else:
        if config.show_warnings:
            print('Unknown ast is found : {} in L.{}'.format(type(astc.nast),astc.lineno))    