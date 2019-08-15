import chainer
import chainer.functions as F
import chainer.links as L
import sys
import numpy as np

import collections

import ast
import gast
import inspect
import six
import types
import weakref
from chainer_compiler.elichika.parser import vevaluator
from chainer_compiler.elichika.parser import core
from chainer_compiler.elichika.parser import nodes
from chainer_compiler.elichika.parser import functions
from chainer_compiler.elichika.parser import utils
from chainer_compiler.elichika.parser import config
from chainer_compiler.elichika.parser import flags

from chainer_compiler.elichika.parser.functions import FunctionBase, UserDefinedFunction

fields = []
histories = []

# hashable function. key is python function, value is FuncValue
function_converters = {}

# unhashable function. key is str, value is FuncValue
builtin_function_converters = {}

# an array of convertter from python instance into Value
# first argument is module, second argument is python instance
instance_converters = []

# assign predefined values
predefined_value_assigners = [] # type: List[PredefinedValueAssigner]

class PredefinedValueAssigner:
    def __init__(self):
        self.target_type = None # type: type

    def assign(self, target : 'Object'):
        return

def apply_predefined_value_assigners(target_type : 'type', target : 'Object'):
    for assigner in predefined_value_assigners:
        if assigner.target_type != target_type:
            continue
        assigner.assign(target)

def create_ref_value_name_with_constant(value):
    if isinstance(value, Object):
        value = value.get_value()

    if value.has_constant_value():
        return '@C_' + str(value.get_constant_value())
    return '@C_Unknown'

def reset_field_and_attributes():
    global fields
    fields = []
    histories.clear()


def register_field(field: 'Field'):
    fields.append(weakref.ref(field))

def unregister_field(field: 'Field'):
    global fields
    fields = [f for f in fields if f() != field]

def push_history(history_id: 'str'):
    histories.append(history_id)
    for field in fields:
        o = field()
        if o is not None:
            o.push_history(history_id)


def pop_history():
    histories.pop()
    for field in fields:
        o = field()
        if o is not None:
            o.pop_history()


def get_inputs() -> 'List[FieldInput]':
    ret = []
    for field in fields:
        o = field()
        if o is not None:
            ret += o.get_inputs()
    return ret


def get_outputs() -> 'List[FieldOutput]':
    ret = []
    for field in fields:
        o = field()
        if o is not None:
            ret += o.get_outputs()
    return ret


def compare(value1, value2):
    if type(value1) != type(value2):
        return False
    else:
        if isinstance(value1, NumberValue):
            return value1.internal_value == value2.internal_value and value1.internal_value is not None
        if isinstance(value1, StrValue):
            return value1.internal_value == value2.internal_value and value1.internal_value is not None
        else:
            return False


def parse_instance(default_module, name, instance, self_instance=None, from_member = False, root_graph : 'graphs.Graph' = None) -> "Object":

    for converter in instance_converters:
        ret = converter(default_module, instance)
        if ret is not None:
            return Object(ret)

    #if inspect.ismethod(instance) or inspect.isfunction(instance) or isinstance(instance, np.ufunc):
    if isinstance(instance, collections.Hashable):
        if instance in function_converters.keys():
            func = function_converters[instance]
            return Object(func)

    # need to check whether is value bool before check whether is value int
    if isinstance(instance, bool):
        return Object(BoolValue(instance))

    if isinstance(instance, int):
        return Object(NumberValue(instance))

    if isinstance(instance, np.int32):
        return Object(NumberValue(instance))

    if isinstance(instance, np.int64):
        return Object(NumberValue(instance))

    if isinstance(instance, float):
        return Object(NumberValue(instance))

    if isinstance(instance, np.float32):
        return Object(NumberValue(instance))

    if isinstance(instance, np.float64):
        return Object(NumberValue(instance))

    if isinstance(instance, str):
        return Object(StrValue(instance))

    if instance is inspect._empty:
        return None

    if inspect.ismethod(instance):
        func = UserDefinedFunction(instance)
        return Object(FuncValue(func, self_instance, default_module))

    if inspect.isfunction(instance):
        func = UserDefinedFunction(instance)
        if from_member:
            return Object(FuncValue(func, self_instance, default_module))
        else:
            return Object(FuncValue(func, None, default_module))

    if inspect.isclass(instance):
        func = functions.UserDefinedClassConstructorFunction(instance)
        return Object(FuncValue(func, None, default_module))

    if isinstance(instance, list):
        if root_graph is None:
            value_in_tuple = []
            for v in instance:
                o = parse_instance(default_module, '', v)
                value_in_tuple.append(o)
            ret = ListValue(value_in_tuple)
        else:
            value_in_tuple = []
            vs = []
            for v in instance:
                o = parse_instance(default_module, '', v)
                value_in_tuple.append(o)
                value = o.get_value()

                if isinstance(value, TupleValue):
                    assert(False)

                if isinstance(value, ListValue):
                    assert(False)

                vs.append(value)

            node = nodes.NodeGenerate('List', vs)
            ret = ListValue(value_in_tuple)
            node.set_outputs([ret])
            root_graph.add_initial_node(node)

        ret.estimate_type()
        return Object(ret)

    if isinstance(instance, dict):
        keys = []
        values = []
        for key, value in instance.items():
            keys.append(parse_instance(default_module, '', key))
            values.append(parse_instance(default_module, '', value))
        ret = DictValue(keys, values)
        return Object(ret)

    if isinstance(instance, tuple) and 'Undefined' in instance:
        shape = list(instance)
        shape = -1 if shape == 'Undefined' else shape
        tensorValue = TensorValue()
        tensorValue.shape = tuple(shape)
        return Object(tensorValue)

    if isinstance(instance, tuple):
        if root_graph is None:
            value_in_tuple = []
            for v in instance:
                o = parse_instance(default_module, '', v)
                value_in_tuple.append(o)

            return Object(TupleValue(value_in_tuple))
        else:
            value_in_tuple = []
            vs = []
            for v in instance:
                o = parse_instance(default_module, '', v)
                value_in_tuple.append(o)
                value = o.get_value()

                if isinstance(value, TupleValue):
                    assert(False)

                if isinstance(value, ListValue):
                    assert(False)

                vs.append(value)

            node = nodes.NodeGenerate('Tuple', vs)
            ret = TupleValue(value_in_tuple)
            node.set_outputs([ret])
            root_graph.add_initial_node(node)
            return Object(ret)


    if isinstance(instance, np.ndarray):
        tensorValue = TensorValue(instance)
        tensorValue.value = instance
        tensorValue.shape = instance.shape
        return Object(tensorValue)

    if isinstance(instance, chainer.Variable):
        tensorValue = TensorValue(instance.data)
        tensorValue.value = instance.data
        tensorValue.shape = instance.data.shape
        return Object(tensorValue)

    if instance == inspect._empty:
        return Object(NoneValue())

    if instance is None:
        return Object(NoneValue())

    if utils.is_disabled_module(instance):
        return None

    if inspect.ismodule(instance):
        value = ModuleValue(instance)
        return Object(value)

    module = Object(ModuleValue(sys.modules[instance.__module__]))
    model_inst = UserDefinedInstance(module, instance, None)
    return Object(model_inst)


class FieldInput:
    def __init__(self):
        self.input_value = None
        self.field = None
        self.name = None
        self.value = None
        self.obj = None

class FieldOutput:
    def __init__(self):
        self.field = None
        self.name = None
        self.obj = None
        self.old_value = None
        self.value = None


class FieldAttributeCollection():
    def __init__(self, id: 'str', parent: 'FieldAttributeCollection'):
        self.id = id
        self.parent = parent
        self.attributes = {}
        self.inputs = {}

    def try_get_attribute(self, key: 'str'):
        if key in self.attributes.keys():
            return self.attributes[key]

        # search from parent
        if self.parent is None:
            return None

        parent_attribute = self.parent.try_get_attribute(key)
        if parent_attribute is None:
            return None

        attribute = Attribute(key)
        attribute.parent = parent_attribute.parent

        # instance or func
        if isinstance(parent_attribute.get_obj().get_value(), Instance) or isinstance(parent_attribute.get_obj().get_value(), FuncValue) or isinstance(parent_attribute.get_obj().get_value(), ModuleValue):
            attribute.revise(parent_attribute.get_obj())
            self.attributes[key] = attribute
            return attribute

        # input
        attribute.revise(parent_attribute.get_obj())
        self.attributes[key] = attribute

        self.inputs[attribute] = (attribute.get_obj(), attribute.get_obj().get_value(
        ), attribute.get_obj().get_value(), attribute.get_obj().get_value())

        return attribute

    def pop_history(self):
        for att, input in self.inputs.items():
            input[0].revise(input[1])
        self.inputs.clear()

    def get_inputs(self) -> 'List[FieldInput]':
        '''
        return [(input value, copied input value)]
        '''
        ret = []
        for att, input in self.inputs.items():
            fi = FieldInput()
            fi.name = att.name
            fi.field = att.parent
            fi.input_value = input[2]
            fi.value = input[3]
            fi.obj = input[0]
            ret.append(fi)
        return ret

    def get_outputs(self) -> 'List[FieldOutput]':
        '''
        return [(field,key,value)]
        '''
        ret = []

        for key, att in self.attributes.items():
            # has ref? (it causes with compile error in almost cases)
            if not att.has_obj():
                continue

            # instance or func
            if isinstance(att.get_obj().get_value(), Instance) or isinstance(att.get_obj().get_value(), FuncValue) or isinstance(att.get_obj().get_value(), ModuleValue):
                continue

            if (not (att in self.inputs.keys())) or att.get_obj() != self.inputs[att][0] or att.get_obj().get_value() != self.inputs[att][1]:
                fo = FieldOutput()
                fo.name = att.name
                fo.field = att.parent
                fo.obj = att.get_obj()
                if att in self.inputs.keys():
                    fo.old_value = self.inputs[att][1]
                fo.value = att.get_obj().get_value()
                ret.append(fo)

        return ret


class Field():
    def __init__(self):
        self.collection = FieldAttributeCollection('', None)
        histories_ = histories.copy()
        histories_.reverse()

        for history in histories_:
            collection = FieldAttributeCollection(history, self.collection)
            self.collection = collection

        self.module = None
        self.id = utils.get_guid()

        register_field(self)

    def dispose(self):
        '''
        dispose this field because of exit function
        don't touch after dispose
        '''
        self.collection = FieldAttributeCollection('', None)
        unregister_field(self)

    def set_module(self, module):
        self.module = module

    def get_field(self) -> 'Field':
        return self

    def has_attribute(self, key) -> 'Boolean':
        c = self.collection

        while c is not None:
            if key in c.attributes.keys():
                return True
            c = c.parent

        return False

    def try_get_attribute(self, key : 'str') -> 'Attribute':
        return self.collection.try_get_attribute(key)

    def get_attribute(self, key: 'str', root_graph : 'graphs.Graph' = None, from_module=False) -> 'Attribute':
        attribute = self.collection.try_get_attribute(key)

        if attribute is not None:
            return attribute

        # search an attribute from a module
        if self.module is not None and from_module and self.module.try_get_and_store_obj(key, root_graph):
            attribute = self.module.attributes.get_attribute(key, root_graph)

        if attribute is not None:
            return attribute

        attribute = Attribute(key)
        attribute.parent = self
        self.collection.attributes[key] = attribute
        return attribute

    def push_history(self, history_id: 'str'):
        collection = FieldAttributeCollection(history_id, self.collection)
        self.collection = collection

    def pop_history(self):
        self.collection.pop_history()
        self.collection = self.collection.parent

        if self.collection is None:
            self.collection = FieldAttributeCollection('', None)

    def get_inputs(self):
        return self.collection.get_inputs()

    def get_outputs(self):
        return self.collection.get_outputs()

    def set_predefined_obj(self, key, obj):
        collections = []
        c = self.collection

        while True:
            collections.append(c)
            c = c.parent
            if c is None:
                break

        collections.reverse()

        old_value = None
        value = None

        for collection in collections:
            attribute = Attribute(key)
            attribute.parent = self
            attribute.revise(obj)
            collection.attributes[key] = attribute

            if isinstance(obj.get_value(), Instance) or isinstance(obj.get_value(), FuncValue) or isinstance(obj.get_value(), ModuleValue):
                continue

            collection.inputs[attribute] = (attribute.get_obj(), attribute.get_obj(
            ).get_value(), attribute.get_obj().get_value(), attribute.get_obj().get_value())

           # if old_value is not None:
           #     collection.inputs[attribute] = (attribute.get_obj(), attribute.get_obj().get_value(), old_value, value)

            #old_value = obj.get_value()
            #value = functions.generate_copied_value(old_value)
            #obj = Object(value)

class Attribute:
    def __init__(self, name: 'str'):
        self.name = name
        self.obj = None
        self.parent = None # type: Field

        # if it is non-volatile, an object in this attribute is saved after running
        self.is_non_volatile = False

    def revise(self, obj: 'Object'):
        assert(isinstance(obj, Object))

        # assgin name to the object
        obj.name = utils.create_obj_value_name_with_attribute(
            self.name, obj.name)
        obj.get_value().name = utils.create_obj_value_name_with_attribute(
            self.name, obj.get_value().name)

        self.obj = obj

    def has_obj(self):
        return self.obj != None

    def get_obj(self):
        assert self.has_obj()
        return self.obj

    def __str__(self):
        return self.name

class Object():
    def __init__(self, value: 'Value'):
        self.name = ""
        self.value = value
        self.id = utils.get_guid()
        self.attributes = Field()
        self.value.apply_to_object(self)
        self.in_container = False

    def get_field(self) -> 'Field':
        return self.attributes

    def get_value(self) -> 'Value':
        return self.value

    def revise(self, value):
        self.value = value

    def try_get_and_store_obj(self, name: 'str', root_graph : 'graphs.Graph') -> 'Object':

        attribute = self.attributes.try_get_attribute(name)
        if attribute is not None and attribute.has_obj():
            return attribute.get_obj()

        obj = self.value.try_get_obj(name, self, root_graph)

        if obj is None:
            return None

        self.attributes.set_predefined_obj(name, obj)
        return obj


class Value():
    def __init__(self):
        self.name = ""
        self.generator = None
        self.internal_value = None
        self.dtype = None
        self.id = utils.get_guid()

        #  this actual value is not important, but type is required as dummy value
        self.is_dummy_value = False

    def has_constant_value(self) -> 'bool':
        return self.internal_value is not None

    def get_constant_value(self):
        return self.internal_value

    def is_not_none_or_any_value(self):
        return False

    def is_iteratable(self):
        return False

    def is_hashable(self):
        return False

    def get_iterator(self) -> 'Object':
        return None

    def apply_to_object(self, obj: 'Object'):
        '''
        register functions to an object
        this function is only called when an object is generated
        '''
        return None

    def encode(self):
        if not self.is_hashable():
            assert(False)
        return ""

    def try_get_obj(self, name: 'str', inst: 'Object', root_graph : 'graphs.Graph') -> 'Object':
        return None

    def __str__(self):
        return self.name


class NoneValue(Value):
    def __init__(self):
        super().__init__()

    def has_constant_value(self) -> 'bool':
        return True

    def is_hashable(self):
        return True

    def encode(self):
        ret = super().encode()
        ret += 'None'
        ret += str(hash(None))
        return ret

    def get_constant_value(self):
        return None

    def __str__(self):
        return self.name + '({})'.format('None')

class UnknownValue(Value):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return self.name + '(Un)'

class NumberValue(Value):
    def __init__(self, number):
        super().__init__()
        self.internal_value = number
        self.dtype = None

        if self.internal_value is not None:
            self.dtype = np.array(self.internal_value).dtype

        if not config.float_restrict and self.dtype == np.float64:
            self.dtype = np.float32

    def is_not_none_or_any_value(self):
        return True

    def is_hashable(self):
        return self.has_constant_value()

    def encode(self):
        ret = super().encode()
        ret += 'Num'
        ret += str(hash(self.internal_value))
        return ret

    def __str__(self):
        if self.internal_value == None:
            return self.name + '(N.{})'.format('Any')
        return self.name + '(N.{})'.format(self.internal_value)


class StrValue(Value):
    def __init__(self, string):
        super().__init__()
        self.internal_value = string

    def is_not_none_or_any_value(self):
        return True

    def is_hashable(self):
        return self.has_constant_value()

    def encode(self):
        ret = super().encode()
        ret += 'Str'
        ret += str(hash(self.internal_value))
        return ret

    def __str__(self):
        if self.internal_value == None:
            return self.name + '(S.{})'.format('Any')
        return self.name + '(S.{})'.format(self.internal_value)


class BoolValue(Value):
    def __init__(self, b):
        super().__init__()
        self.internal_value = b

    def is_not_none_or_any_value(self):
        return True

    def is_hashable(self):
        return self.has_constant_value()

    def encode(self):
        ret = super().encode()
        ret += 'Num'
        ret += str(hash(self.internal_value))
        return ret

    def __str__(self):
        if self.internal_value == None:
            return self.name + '(B.{})'.format('Any')
        return self.name + '(B.{})'.format(self.internal_value)


class RangeValue(Value):
    def __init__(self):
        super().__init__()

    def is_not_none_or_any_value(self):
        return True

    def is_iteratable(self):
        return True

    def get_iterator(self) -> 'Object':
        return Object(NumberValue(None))

    def __str__(self):
        return self.name + '(R)'


class TupleValue(Value):
    def __init__(self, values=None):
        super().__init__()
        self.internal_value = values
        self.vtype = None # type: Type

    def is_not_none_or_any_value(self):
        return True

    def is_iteratable(self):
        return True

    def is_hashable(self):
        self.estimate_type()
        return self.has_constant_value() and self.vtype is not None

    def encode(self):
        ret = super().encode()
        ret += 'Tuple'
        tup = tuple(v.get_value().internal_value for v in self.internal_value)
        ret += str(hash(tup))
        return ret

    def get_iterator(self) -> 'Object':
        if self.vtype is None:
            return None

        v = self.vtype()

        if self.dtype is not None:
            v.dtype = self.dtype


        return Object(v)

    def estimate_type(self):
        if self.internal_value is None:
            return

        self.vtype = None
        self.dtype = None

        for v in self.internal_value:
            if self.vtype is None:
                self.vtype = type(v.get_value())
                self.dtype = v.get_value().dtype
            else:
                if self.vtype != type(v.get_value()):
                    self.vtype = None
                    self.dtype = None
                    return
                if self.dtype != v.get_value().dtype:
                    self.dtype = None

    def __str__(self):
        return self.name + '(Tp{})'


class FuncValue(Value):
    def __init__(self, func: 'functions.FunctionBase', obj: 'Object', module : 'Object' = None):
        super().__init__()
        self.func = func
        self.internal_value = func  # TODO(rchours): So that has_constant_type() succeeds on FuncValue.
        self.obj = obj
        self.module = module

    def is_not_none_or_any_value(self):
        return True

    def __str__(self):
        return self.name + '(F)'


class ListValue(Value):
    def __init__(self, values=None):
        super().__init__()
        self.internal_value = values
        self.dtype = None
        self.vtype = None # type: Type

    def is_not_none_or_any_value(self):
        return True

    def is_iteratable(self):
        return True

    def get_iterator(self) -> 'Object':
        if self.vtype is None:
            return None

        v = self.vtype()

        if self.dtype is not None:
            v.dtype = self.dtype


        return Object(v)

    def __filter_internal_values(self):
        return [v for v in self.internal_value if v is not None and not isinstance(v.get_value(), NoneValue)]

    def estimate_type(self):
        if self.internal_value is None:
            return

        internal_values = self.__filter_internal_values()

        self.vtype = None
        self.dtype = None

        for v in internal_values:
            if self.vtype is None:
                self.vtype = type(v.get_value())
                self.dtype = v.get_value().dtype
            else:
                if self.vtype != type(v.get_value()):
                    self.vtype = None
                    self.dtype = None
                    return
                if self.dtype != v.get_value().dtype:
                    self.dtype = None

    def append(self, v):
        if self.internal_value is None:
            if self.vtype is None and not isinstance(v.get_value(), NoneValue):
                self.vtype = type(v.get_value())
                self.dtype = v.get_value().dtype
            else:
                if self.vtype != type(v.get_value()):
                    self.vtype = None
                    self.dtype = None
                    return
                if self.dtype != v.get_value().dtype:
                    self.dtype = None

        else:
            self.internal_value.append(v)
            self.estimate_type()

    def apply_to_object(self, obj: 'Object'):
        apply_predefined_value_assigners(type(ListValue), obj)

    def __str__(self):
        return self.name + '(L)'

class DictValue(Value):
    def __init__(self, keys=None, values=None):
        super().__init__()
        self.internal_keys = {}
        self.internal_values = Field()
        self.key_dtype = None
        self.key_vtype = None # type: Type

        for key, value in zip(keys, values):
            if key.get_value().is_hashable():
                key_hash = key.get_value().encode()
                self.internal_values.get_attribute(key_hash).revise(value)
                self.internal_keys[key_hash] = key
            else:
                assert False  # Non hashable types not supported

    def is_not_none_or_any_value(self):
        return True

    def is_iteratable(self):
        return False

    # TODO(rchouras): Add iterator for dictionary keys.
    # def get_iterator(self) -> 'Object':
    #     return

    def apply_to_object(self, obj: 'Object'):
        apply_predefined_value_assigners(type(ListValue), obj)
        '''
        keys_func = Object(
            FuncValue(functions_builtin.KeysFunction(self), obj, None))
        obj.attributes.get_attribute('keys').revise(keys_func)

        values_func = Object(
            FuncValue(functions_builtin.ValuesFunction(self), obj, None))
        obj.attributes.get_attribute('values').revise(values_func)
        '''
    def __str__(self):
        return self.name + '(D)'

class TensorValue(Value):
    def __init__(self, value = None):
        super().__init__()
        self.shape = ()
        self.internal_value = value
        self.value = None   # not used?
        self.dtype = None

        if self.internal_value is not None:
            self.dtype = np.array(self.internal_value).dtype

        if not config.float_restrict and self.dtype == np.float64:
            self.dtype = np.float32

    def is_not_none_or_any_value(self):
        return True

    def is_iteratable(self):
        return True

    def get_iterator(self) -> 'Object':
        v = TensorValue()
        v.dtype = self.dtype
        return Object(v)

    def apply_to_object(self, obj: 'Object'):
        apply_predefined_value_assigners(type(TensorValue), obj)

    def __str__(self):
        return self.name + '(T.{})'.format(self.shape)


class Type(Value):
    def __init__(self, name: 'str'):
        super().__init__()
        self.name = name

    def is_not_none_or_any_value(self):
        return True

class ModuleValue(Value):
    def __init__(self, module):
        super().__init__()
        self.internal_module = module


    def try_get_obj(self, name: 'str', inst: 'Object', root_graph : 'graphs.Graph') -> 'Object':


        if self.internal_module == six.moves:
            # Calling `inspect.getmembers` for `six.moves` causes
            # eager load for potentially non-existent libraries such
            # as tkinter or gdbm. To workaround this issue, we
            # retrieve only whitelisted members in `six.moves`.
            # TODO(hamaji): Figure out a better workaround.
            safe_keys = ['range', 'xrange', 'map', 'filter', 'zip']
            members = [(k, getattr(self.internal_module, k)) for k in safe_keys]
        else:
            members = inspect.getmembers(self.internal_module)
        members_dict = {}
        for member in members:
            members_dict[member[0]] = member[1]

        if not (name in members_dict.keys()):
            if name in builtin_function_converters.keys():
                v = Object(builtin_function_converters[name])
                return v
            return None

        attr_v = members_dict[name]

        dummy_flags_members_dict = {}
        dummy_flags_members = inspect.getmembers(flags)
        for member in dummy_flags_members:
            if isinstance(member[1], types.FunctionType):
                dummy_flags_members_dict[member[0]] = member[1]

        if name in dummy_flags_members_dict.keys():
            v = Object(builtin_function_converters[name])
            return v

        v = parse_instance(inst, name, attr_v, None)
        return v

class Instance(Value):
    def __init__(self, module: 'Object', inst, classinfo):
        super().__init__()
        self.inst = inst
        self.func = None
        self.module = module
        self.classinfo = classinfo

    def is_not_none_or_any_value(self):
        return True

class UserDefinedInstance(Instance):
    def __init__(self, module: 'Object', inst, classinfo):
        super().__init__(module, inst, classinfo)

    def try_get_obj(self, name: 'str', inst: 'Object', root_graph : 'graphs.Graph') -> 'Object':
        obj = None
        if self.inst is not None:
            if not hasattr(self.inst, name):
                return None

            attr_v = getattr(self.inst, name)
            obj = parse_instance(self.module, name, attr_v, inst, root_graph=root_graph)

        else:
            members = inspect.getmembers(self.classinfo)
            members_dict = {}
            for member in members:
                members_dict[member[0]] = member[1]

            if not (name in members_dict.keys()):
                return None

            obj = parse_instance(self.module, name, members_dict[name], inst, from_member=True, root_graph=root_graph)

        return obj

    def apply_to_object(self, obj: 'values.Object'):
        super().apply_to_object(obj)

        enter_func = obj.try_get_and_store_obj('__enter__', None)
        if enter_func is not None:
            obj.get_field().get_attribute('__enter__').revise(enter_func)

        exit_func = obj.try_get_and_store_obj('__exit__', None)
        if exit_func is not None:
            obj.get_field().get_attribute('__exit__').revise(exit_func)

        getitem_func = obj.try_get_and_store_obj('__getitem__', None)
        if getitem_func is not None:
            obj.get_field().get_attribute('__getitem__').revise(getitem_func)
