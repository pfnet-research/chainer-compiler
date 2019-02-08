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

def try_get_obj(value, name, lineprop) -> 'values.Object':
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
            return value.get_obj()

    if isinstance(value, values.Object):
        return value

    return None

def try_get_value(value, name, lineprop, is_none_allowed = False) -> 'values.Value':
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

    if isinstance(value, values.Object):
        return value.get_value()

    if isinstance(value, values.Attribute):
        return value.get_obj().get_value()

    raise Exception('Value {} is invalid. in L.{}'.format(name, lineprop))

def get_ast_name_forcibly(ast):
    if isinstance(ast, gast.gast.Name):
        return ast.id
    if isinstance(ast, gast.gast.Attribute):
        return ast.attr
    return ''

def get_input_attritubtes(target : 'values.Field', commit_id1 : 'str', commit_id2 : 'str'):
    ret = [v for v in target.attributes.values() if v.has_accessed(commit_id1, commit_id2)]
    dicts = [v.get_obj(False) for v in target.attributes.values()]

    for d in dicts:
        ret_ = get_input_attritubtes(d.get_field(), commit_id1, commit_id2)
        ret.extend(ret_)

    return ret

def get_output_attritubtes(target : 'values.Field', commit_id1 : 'str', commit_id2 : 'str'):
    ret = [v for v in target.attributes.values() if v.has_diff(commit_id1, commit_id2)]
    dicts = [v.get_obj(False) for v in target.attributes.values()]

    for d in dicts:
        ret_ = get_output_attritubtes(d.get_field(), commit_id1, commit_id2)
        ret.extend(ret_)

    return ret

def get_output_objs(target : 'values.Field', commit_id1 : 'str', commit_id2 : 'str'):
    ret = [v.get_obj(False) for v in target.attributes.values() if v.get_obj(False).has_diff(commit_id1, commit_id2)]
    dicts = [v.get_obj(False) for v in target.attributes.values()]

    for d in dicts:
        ret_ = get_output_objs(d.get_field(), commit_id1, commit_id2)
        ret.extend(ret_)

    return ret

def filter_attributes(attributes):
    ret = []

    for attribute in attributes:
        if attribute.has_obj() and isinstance(attribute.get_obj(False).get_value(), values.Instance):
            continue

        if attribute.has_obj() and isinstance(attribute.get_obj(False).get_value(), values.FuncValue):
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
    lineprop = utils.LineProperty(astc.lineno)

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    value_obj = try_get_obj(value, 'attribute', lineprop)
    attr = value_obj.get_field().get_attribute(astc.nast.attr)

    if attr.has_obj():
        return attr

    # if attr is not found
    gotten_obj = value_obj.try_get_and_store_obj(astc.nast.attr)
    if gotten_obj is not None:
        return attr

    return attr

def veval_ast_assign(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.Assign))
    lineprop = utils.LineProperty(astc.lineno)

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    value_obj = try_get_obj(value, 'assign', lineprop)

    if value is None:
        if config.show_warnings:
            print('It is possible that assiging value is invalid in L.{}'.format(astc.lineno))
        return None

    targets = veval_ast(astc.c(astc.nast.targets[0]), local_field, graph)

    isTuple = False
    if targets.has_obj() and isinstance(targets.get_obj().get_value(), values.TupleValue):
        targets = targets.get_obj().values
        isTuple = True

    if isTuple:
        for i in range(len(targets)):
            node_assign = nodes.NodeAssign(targets[i], value.values[i], astc.lineno)
            targets[i].revise(value.values[i])
            graph.add_node(node_assign)
    else:
        node_assign = nodes.NodeAssign(targets, value_obj, astc.lineno)
        targets.revise(value_obj)
        graph.add_node(node_assign)

def veval_ast_name(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Name))

    ret = local_field.get_attribute(astc.nast.id)
    return ret

def veval_ast_call(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'Attribute':
    assert(isinstance(astc.nast, gast.gast.Call))
    lineprop = utils.LineProperty(astc.lineno)

    func = veval_ast(astc.c(astc.nast.func), local_field, graph)
    if func == None or not func.has_obj():
        if config.show_warnings:
            print('Unknown function {} is called in L.{}'.format(get_ast_name_forcibly(astc.nast.func) ,astc.lineno))
        return None

    func_obj = try_get_obj(func, 'call', lineprop)
    func_value = try_get_value(func, 'call', lineprop)

    fargs = []
    for arg in astc.nast.args:
        arg_ = veval_ast(astc.c(arg), local_field, graph)
        farg = functions.FunctionArg()
        farg.obj = try_get_obj(arg_, 'call', lineprop)
        fargs.append(farg)

    for keyword in astc.nast.keywords:
        arg_ = veval_ast(astc.c(keyword.value), local_field, graph)
        farg = functions.FunctionArg()
        farg.name = keyword.arg
        farg.obj = try_get_obj(arg_, 'call', lineprop)
        fargs.append(farg)

    lineprop = utils.LineProperty(astc.lineno)

    ret = None
    if isinstance(func_value, values.FuncValue):
        ret = func_value.func.vcall(local_field.module, graph, func_value.obj, fargs, lineprop)

    elif isinstance(func_value, values.Instance) and func_value.callable:
        # __call__
        ret = func_value.func.get_value().func.vcall(local_field.module, graph, func_obj, fargs, lineprop)

    else:
        if config.show_warnings:
            print('Unknown function is called in L.{}'.format(astc.lineno))
        return None

    return ret

def veval_ast_return(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph') -> 'None':
    assert(isinstance(astc.nast, gast.gast.Return))
    lineprop = utils.LineProperty(astc.lineno)

    value = veval_ast(astc.c(astc.nast.value), local_field, graph)
    value_obj = try_get_obj(value, 'return', lineprop)
    value_value = try_get_value(value, 'return', lineprop)

    if value is values.Attribute and not value.has_obj():
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

    values.commit(if_id)

    # True condition
    values.checkout(if_id)
    true_graph = Graph()
    true_graph.name = 'True'
    body = veval_ast(astc.c(astc.nast.body), local_field, true_graph)

    values.commit(true_id)
    true_input_attributes = get_input_attritubtes(local_field, if_id, true_id)
    true_output_attributes = get_output_attritubtes(local_field, if_id, true_id)
    true_output_objs = get_output_objs(local_field, if_id, true_id)

    #TODO(durswd): improve
    true_input_attributes = filter_attributes(true_input_attributes)
    true_output_attributes = filter_attributes(true_output_attributes)

    true_output_attributes_2_values = {}

    for attribute in true_output_attributes:
        true_output_attributes_2_values[attribute] = attribute.get_obj().get_value()

    # False condition
    values.checkout(if_id)
    false_graph = Graph()
    false_graph.name = 'False'
    orelse = veval_ast(astc.c(astc.nast.orelse), local_field, false_graph)

    values.commit(false_id)
    false_input_attributes = get_input_attritubtes(local_field, if_id, false_id)
    false_output_attributes = get_output_attritubtes(local_field, if_id, false_id)
    false_output_objs = get_output_objs(local_field, if_id, false_id)

    #TODO(durswd): improve
    false_input_attributes = filter_attributes(false_input_attributes)
    false_output_attributes = filter_attributes(false_output_attributes)

    false_output_attributes_2_values = {}

    for attribute in false_output_attributes:
        false_output_attributes_2_values[attribute] = attribute.get_obj().get_value()

    # Merge
    values.checkout(if_id)

    # Input
    input_attributes = set(true_input_attributes) | set(false_input_attributes)

    # remove unexisting values
    input_attributes = [v for v in input_attributes if v.has_obj()]
    input_values = [i.get_obj().get_value() for i in input_attributes]

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

    obj2output_values = {}
    for obj in true_output_objs:
        if obj.get_value() == None:
            continue

        key = obj
        value = obj.get_value_log(true_id)

        if key in obj2output_values.keys():
            obj2output_values[key][0] = value
        else:
            obj2output_values[key] = [value, None]

    for obj in false_output_objs:
        if obj.get_value() == None:
            continue

        key = obj
        value = obj.get_value_log(false_id)

        if key in obj2output_values.keys():
            obj2output_values[key][1] = value
        else:
            obj2output_values[key] = [None, value]


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
                true_value = parent.get_attribute(name).get_obj().get_value()

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
                false_value = parent.get_attribute(name).get_obj().get_value()

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
            parent.get_attribute(name).revise(values.Object(value))

        elif true_attribute is not None and false_attribute is not None:
            # change both
            value = functions.generate_value_with_same_type(true_value)
            output_values.append(value)

            if parent.get_attribute(name).is_non_volatile:
                non_volatiles.append((parent.get_attribute(name).initial_obj.get_value(), value))

            parent.get_attribute(name).revise(values.Object(value))

        elif true_attribute in input_attributes:
            value = functions.generate_value_with_same_type(true_value)

            if parent.get_attribute(name).is_non_volatile:
                non_volatiles.append((parent.get_attribute(name).initial_obj.get_value(), value))

            output_values.append(value)
            parent.get_attribute(name).revise(values.Object(value))
        else:
            value = functions.generate_value_with_same_type(false_value)

            if parent.get_attribute(name).is_non_volatile:
                non_volatiles.append((parent.get_attribute(name).initial_obj.get_value(), value))

            output_values.append(value)
            parent.get_attribute(name).revise(values.Object(value))

    for input_value in input_values:
        true_graph.add_input_value(input_value)
        false_graph.add_input_value(input_value)

    for obj, values_pairs in obj2output_values.items():
        if not obj.get_value() in input_values:
            input_values.append(obj.get_value())

        value = None
        true_value = None
        false_value = None

        if values_pairs[0] is not None:
            value = values_pairs[0]
            true_value = values_pairs[0]
        if values_pairs[1] is not None:
            value = values_pairs[1]
            false_value = values_pairs[1]

        if true_value is None:
            true_value = obj.get_value()

        if false_value is None:
            false_value = obj.get_value()

        value = functions.generate_value_with_same_type(value)
        obj.revise(value)
        output_values.append(value)

        true_graph.add_output_value(true_value)
        false_graph.add_output_value(false_value)


    node = nodes.NodeIf(test_value, input_values, true_graph, false_graph, astc.lineno)
    node.set_outputs(output_values)

    graph.add_node(node)

    # add non-volatiles
    for tv, v in non_volatiles:
        node_nv = nodes.NodeNonVolatileAssign(tv, v)
        graph.add_node(node_nv)

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
    
    new_value = functions.generate_value_with_same_type(target_value)
    node_aug_assign.set_outputs([new_value])
    target.get_obj().revise(new_value)
        
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
            return values.NumberValue(default_value)
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
            values_ = [try_get_value(x, 'subscript', lineprop) for x in slice_value.values]
            node = nodes.NodeGetItem(value_value, values_)
        else:
            # ex. x[1]
            node = nodes.NodeGetItem(value_value, [slice_value])
        ret_value = values.Value()
        node.set_outputs([ret_value])
        graph.add_node(node)
        return values.Object(ret_value)

    elif isinstance(astc.nast.slice, gast.gast.Slice):

        indices = get_slice_indices(astc.nast.slice)

        node = nodes.NodeSlice(value_value, indices, [len(indices)])
        ret_value = values.Value()
        node.set_outputs([ret_value])
        graph.add_node(node)
        return values.Object(ret_value)

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
        ret_value = values.Value()
        node.set_outputs([ret_value])
        graph.add_node(node)
        return values.Object(ret_value)

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
    internal_iter_id = '@internal/iter_' + listcomp_guid

    generator = astc.nast.generators[0]
    iter_value = try_get_value(veval_ast(astc.c(generator.iter), local_field, graph), 'generator', lineprop)
    list_value = values.ListValue()
    list_obj = values.Object(list_value)

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

    counter_value = values.NumberValue(0)
    counter_value.name = 'listcomp_counter_' + listcomp_guid

    cond_value = values.BoolValue(True)
    cond_value.name = 'listcomp_cond_' + listcomp_guid

    body_field = values.Field()
    body_field.set_module(local_field.module)
    body_field.set_parent(local_field)

    # set iter with internal name
    body_field.get_attribute(internal_iter_id).revise(list_obj)

    values.commit(listcomp_id)

    body_graph = Graph()
    body_graph.name = 'Body_' + listcomp_guid

    node_forgen = nodes.NodeForGenerator(counter_value, iter_value)
    target_value = values.Value()
    target_obj = values.Object(target_value)
    node_forgen.set_outputs([target_value])
    
    body_field.get_attribute(target_name).revise(target_obj)

    body_graph.add_node(node_forgen)

    elt = veval_ast(astc.c(astc.nast.elt), body_field, body_graph)
    elt_obj = try_get_obj(elt, 'listcomp', lineprop)

    farg = functions.FunctionArg()
    farg.name = ''
    farg.obj = elt_obj
    append_value = list_obj.get_field().get_attribute('append').get_obj().get_value()
    append_value.func.vcall(local_field.module, body_graph, list_obj, [farg], lineprop)

    values.commit(body_id)

    body_input_attributes = get_input_attritubtes(body_field, listcomp_id, body_id) + get_input_attritubtes(local_field, listcomp_id, body_id)
    body_output_attributes = get_output_attritubtes(body_field, listcomp_id, body_id) + get_output_attritubtes(local_field, listcomp_id, body_id)
    body_input_attributes = filter_attributes(body_input_attributes)
    body_output_attributes = filter_attributes(body_output_attributes)

    output_objs = get_output_objs(local_field, listcomp_id, body_id) + get_output_objs(body_field, listcomp_id, body_id)

    output_attributes_2_values = {}

    for attribute in body_output_attributes:
        output_attributes_2_values[attribute] = attribute.get_obj().get_value()

    # get inputs
    values.checkout(listcomp_id)

    input_attributes_2_values = {}

    for attribute in body_input_attributes:
        input_attributes_2_values[attribute] = attribute.get_obj().get_value()

    # Exports
    values.checkout(listcomp_id)

    # generate attribute pairs
    name2attributes = {}

    for attribute in body_input_attributes:
        key = str(attribute.parent.id) + '_' + attribute.name

        if key in name2attributes.keys():
            name2attributes[key][0] = attribute
        else:
            name2attributes[key] = [attribute, None]

    for attribute in body_output_attributes:
        key = str(attribute.parent.id) + '_' + attribute.name

        if key in name2attributes.keys():
            name2attributes[key][1] = attribute
        else:
            name2attributes[key] = [None, attribute]

    # remove defaule values
    name_removing = []
    defaule_values = [counter_value, cond_value, iter_value]
    for k, v in name2attributes.items():
        if v[0] is not None and input_attributes_2_values[v[0]] in defaule_values:
            name_removing.append(k)

        if v[1] is not None and output_attributes_2_values[v[1]] in defaule_values:
            name_removing.append(k)

    for nr in name_removing:
        if nr in name2attributes.keys():
            name2attributes.pop(nr)

    #
    inputs = []
    outputs = []
    non_volatiles = []

    # default input for subgraph's input
    body_graph.add_input_value(counter_value)
    body_graph.add_input_value(cond_value)
    body_graph.add_input_value(iter_value)

    # default output for subgrap's output
    body_graph.add_output_value(cond_value)
    body_graph.add_output_value(iter_value)

    # default output
    outputs.append(functions.generate_value_with_same_type(iter_value))
    
    for attributes in name2attributes.values():
        name = ''
        parent = None
        input_value = None
        output_value = None

        if attributes[0] is not None:
            name = attributes[0].name
            parent = attributes[0].parent

        if attributes[1] is not None:
            name = attributes[1].name
            parent = attributes[1].parent

        if attributes[0] is not None:
            input_value = input_attributes_2_values[attributes[0]]
        else:
            # value with same type
            input_value = functions.generate_value_with_same_type(output_attributes_2_values[attributes[1]])

        if attributes[1] is not None:
            output_value = output_attributes_2_values[attributes[1]]
        else:
            # copy value
            output_value = input_attributes_2_values[attributes[0]]

        output_value_in_node = functions.generate_value_with_same_type(output_value)

        inputs.append(input_value)
        outputs.append(output_value_in_node)
        body_graph.add_input_value(input_value)
        body_graph.add_output_value(output_value)

        if attributes[1] is not None and attributes[1].is_non_volatile:
            non_volatiles.append((attribute[1].initial_obj.get_value(),output_value_in_node))

        output_obj_in_node = values.Object(output_value_in_node)
        parent.get_attribute(name).revise(output_obj_in_node)

    for obj in output_objs:
        if obj.get_value() is None:
            continue
        inputs.append(obj.get_value())
        body_graph.add_input_value(obj.get_value())
        value = obj.get_value_log(body_id)
        body_graph.add_output_value(value)
        value = functions.generate_value_with_same_type(value)
        obj.revise(value)
        outputs.append(value)

    node = nodes.NodeListcomp(iter_value, inputs, body_graph, astc.lineno)
    node.set_outputs(outputs)

    graph.add_node(node)

    # add non-volatiles
    for tv, v in non_volatiles:
        node_nv = nodes.NodeNonVolatileAssign(tv, v)
        graph.add_node(node_nv)

    if body_field.get_attribute(internal_iter_id).has_obj():
        return body_field.get_attribute(internal_iter_id).get_obj()
    else:
        return list_obj

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

    return values.Object(ret_value)

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

    return values.Object(ret_value)


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
    node_compare.set_outputs([ret_value])
    graph.add_node(node_compare)

    return values.Object(ret_value)


def veval_ast_num(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. 1, 2, ...
    '''
    assert(isinstance(astc.nast, gast.gast.Num))
    lineprop = utils.LineProperty(astc.lineno)
    value = values.NumberValue(astc.nast.n)
    return values.Object(value)

def veval_ast_name_constant(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    Ex. True
    '''
    assert(isinstance(astc.nast, gast.gast.NameConstant))
    lineprop = utils.LineProperty(astc.lineno)
    if astc.nast.value == True:
        return values.Object(values.BoolValue(True))
    if astc.nast.value == False:
        return values.Object(values.BoolValue(False))
    if astc.nast.value is None:
        return values.Object(values.BoolValue(False))
    return None

def veval_ast_tuple(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    assert(isinstance(astc.nast, gast.gast.Tuple))
    vs = []
    for v in astc.nast.elts:
        v_ = veval_ast(astc.c(v), local_field, graph)
        vs.append(v_)

    return values.Object(values.TupleValue(vs))

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
        elt_obj = try_get_obj(elt_,'list', lineprop)
        elts.append(elt_obj)

    node = nodes.NodeGenerate('List', [elt.get_value() for elt in elts], lineprop)
    graph.add_node(node)
    value = values.ListValue()
    value.values.extend(elts)
    node.set_outputs([value])

    return values.Object(value)

def veval_ast_for(astc : 'AstContext', local_field : 'values.Field', graph : 'Graph'):
    '''
    for target in iter:
        ...
    '''
    assert(isinstance(astc.nast, gast.gast.For))
    lineprop = utils.LineProperty(astc.lineno)

    # for target in iter:
    iter_ = veval_ast(astc.c(astc.nast.iter), local_field, graph)

    # get target name
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
    body_graph = Graph()
    body_graph.name = 'Body'

    counter_value = values.NumberValue(0)
    counter_value.name = 'for_counter'

    cond_value = values.BoolValue(True)
    cond_value.name = 'for_cond'

    iter_value = try_get_value(iter_, 'for', lineprop)

    # node to lookup a value from sequence
    node_forgen = nodes.NodeForGenerator(counter_value, iter_value)
    target_value = values.Value()
    target_obj = values.Object(target_value)
    node_forgen.set_outputs([target_value])

    target_attribute = local_field.get_attribute(target_name)
    target_attribute.revise(target_obj)
    body_graph.add_node(node_forgen)

    body = veval_ast(astc.c(astc.nast.body), local_field, body_graph)

    values.commit(body_id)

    body_input_attributes = get_input_attritubtes(local_field, for_id, body_id)
    body_output_attributes = get_output_attritubtes(local_field, for_id, body_id)

    body_input_attributes = filter_attributes(body_input_attributes)
    body_output_attributes = filter_attributes(body_output_attributes)

    output_attributes_2_values = {}

    for attribute in body_output_attributes:
        output_attributes_2_values[attribute] = attribute.get_obj().get_value()

    output_objs = get_output_objs(local_field, for_id, body_id)

    # get inputs
    values.checkout(for_id)

    input_attributes_2_values = {}

    for attribute in body_input_attributes:
        input_attributes_2_values[attribute] = attribute.get_obj().get_value()

    # Exports
    values.checkout(for_id)

    # generate attribute pairs
    name2attributes = {}

    for attribute in body_input_attributes:
        key = str(attribute.parent.id) + '_' + attribute.name

        if key in name2attributes.keys():
            name2attributes[key][0] = attribute
        else:
            name2attributes[key] = [attribute, None]

    for attribute in body_output_attributes:
        key = str(attribute.parent.id) + '_' + attribute.name

        if key in name2attributes.keys():
            name2attributes[key][1] = attribute
        else:
            name2attributes[key] = [None, attribute]

    # remove defaule values
    name_removing = []
    defaule_values = [counter_value, cond_value, iter_value]
    for k, v in name2attributes.items():
        if v[0] in defaule_values:
            name_removing.append(k)

        if v[1] in defaule_values:
            name_removing.append(k)

    for nr in name_removing:
        if nr in name2attributes.keys():
            name2attributes.pop(nr)

    #
    inputs = []
    outputs = []
    non_volatiles = []

    # default input for subgraph's input
    body_graph.add_input_value(counter_value)
    body_graph.add_input_value(cond_value)
    body_graph.add_input_value(iter_value)

    # default output for subgrap's output
    body_graph.add_output_value(cond_value)
    body_graph.add_output_value(iter_value)

    # default output
    outputs.append(functions.generate_value_with_same_type(iter_value))
    
    for attributes in name2attributes.values():
        name = ''
        parent = None
        input_value = None
        output_value = None

        if attributes[0] is not None:
            name = attributes[0].name
            parent = attributes[0].parent

        if attributes[1] is not None:
            name = attributes[1].name
            parent = attributes[1].parent

        if attributes[0] is not None:
            input_value = input_attributes_2_values[attributes[0]]
        else:
            # value with same type
            input_value = functions.generate_value_with_same_type(output_attributes_2_values[attributes[1]])

        if attributes[1] is not None:
            output_value = output_attributes_2_values[attributes[1]]
        else:
            # copy value
            output_value = input_attributes_2_values[attributes[0]]

        output_value_in_node = functions.generate_value_with_same_type(output_value)

        inputs.append(input_value)
        outputs.append(output_value_in_node)
        body_graph.add_input_value(input_value)
        body_graph.add_output_value(output_value)

        if attributes[1].is_non_volatile:
            non_volatiles.append((attributes[1].initial_obj.get_value(),output_value_in_node))

        output_obj_in_node = values.Object(output_value_in_node)
        attributes[1].parent.get_attribute(name).revise(output_obj_in_node)

    for obj in output_objs:
        if obj.get_value() is None:
            continue
        value = obj.get_value_log(body_id)
        body_graph.add_output_value(value)
        value = functions.generate_value_with_same_type(value)
        obj.revise(value)
        outputs.append(value)

    node = nodes.NodeFor(iter_value, inputs, body_graph, astc.lineno)
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
        return veval_ast_listcomp(astc, local_field, graph)

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
