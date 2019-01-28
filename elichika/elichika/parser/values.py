import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

import inspect
import ast, gast
import weakref
from elichika.parser import vevaluator
from elichika.parser import core
from elichika.parser import nodes
from elichika.parser import functions
from elichika.parser import utils
from elichika.parser import config
from elichika.parser import functions_builtin

from elichika.parser.functions import FunctionBase, UserDefinedFunction

fields = []
attributes = []
registered_values = []

def reset_field_and_attributes():
    global fields
    global attributes
    global modified_values
    fields = []
    attributes = []
    modified_values = []

def register_field(field : 'Field'):
    fields.append(weakref.ref(field))

def register_attribute(attribute : 'Attribute'):
    attributes.append(weakref.ref(attribute))

def register_value(value : 'Value'):
    registered_values.append(weakref.ref(value))

def commit(commit_id : 'str'):
    for field in fields:
        o = field()
        if o is not None:
            o.commit(commit_id)

    for attribute in attributes:
        o = attribute()
        if o is not None:
            o.commit(commit_id)

    for registered_value in registered_values:
        o = registered_value()
        if o is not None:
            o.commit(commit_id)

def checkout(commit_id : 'str'):
    for field in fields:
        o = field()
        if o is not None:
            o.checkout(commit_id)

    for attribute in attributes:
        o = attribute()
        if o is not None:
            o.checkout(commit_id)

    for registered_value in registered_values:
        o = registered_value()
        if o is not None:
            o.checkout(commit_id)

def parse_instance(default_module, name, instance, self_instance = None):
    from elichika.parser import values_builtin

    if values_builtin.is_builtin_chainer_link(instance):
        return values_builtin.ChainerLinkInstance(default_module, instance)

    # need to check whether is value bool before check whether is value int
    if isinstance(instance, bool):
        return BoolValue(instance)

    if isinstance(instance, int) or isinstance(instance, float):
        return NumberValue(instance)

    if isinstance(instance, str):
        return StrValue(instance)

    if isinstance(instance, list):
        ret = ListValue()
        ind = 0
        for e in instance:
            element_value = parse_instance(default_module, '', e)
            ret.get_field().get_attribute(str(ind)).revise(element_value)
            ind += 1
        return ret

    if instance is inspect._empty:
        return None

    if inspect.isfunction(instance):
        func = UserDefinedFunction(instance)
        return FuncValue(func, self_instance)

    if inspect.ismethod(instance):
        func = UserDefinedFunction(instance)
        return FuncValue(func, self_instance)

    if inspect.isclass(instance):
        func = functions.UserDefinedClassConstructorFunction(instance)
        return FuncValue(func, None)

    if isinstance(instance, tuple) and 'Undefined' in instance:
        shape = list(instance)
        shape = -1 if shape == 'Undefined' else shape
        tensorValue = TensorValue()
        tensorValue.shape = tuple(shape)
        return tensorValue

    if isinstance(instance, np.ndarray):
        tensorValue = TensorValue()
        tensorValue.value = instance
        tensorValue.shape = instance.shape
        return tensorValue

    if instance == inspect._empty:
        return NoneValue()

    if instance is None:
        return NoneValue()

    model_inst = UserDefinedInstance(default_module, instance, None, isinstance(instance, chainer.Link))
    return model_inst

class Field():
    def __init__(self):
        self.attributes = {}
        self.module = None
        self.parent = None

        self.rev_attributes = {}
        self.id = utils.get_guid()

        register_field(self)

    def set_module(self, module):
        self.module = module

    def set_parent(self, parent):
        self.parent = parent

    def get_field(self) -> 'Field':
        return self

    def has_attribute(self, key) -> 'Boolean':

        if key in self.attributes.keys():
            return True

        return False

    def get_attribute(self, key : 'str') -> 'Attribute':
        if key in self.attributes.keys():
            return self.attributes[key]
        else:
            # search an attribute from parents
            attribute = None
            if self.parent is not None and self.parent.has_attribute(key):
                attribute = self.parent.get_attribute(key)

            if attribute is not None:
                return attribute

            # search an attribute from a module
            if self.module is not None and self.module.has_attribute(key):
                attribute = self.module.get_attribute(key)

            if attribute is not None:
                return attribute

            attribute = Attribute(key)
            attribute.parent = self
            self.attributes[key] = attribute
            return attribute

    def commit(self, commit_id : 'str'):
        self.rev_attributes[commit_id] = self.attributes.copy()

    def checkout(self, commit_id : 'str'):
        if commit_id in self.rev_attributes:
            self.attributes = self.rev_attributes[commit_id].copy()
        else:
            self.attributes = {}

    def set_default_value(self, key, value):
        attribute = self.get_attribute(key)
        attribute.revise(value)

class Module(Field):
    def __init__(self, module):
        super().__init__()
        self.internal_module = module

    def has_attribute(self, key) -> 'Boolean':
        members = inspect.getmembers(self.internal_module)
        members_dict = {}
        for member in members:
            members_dict[member[0]] = member[1]

        if (key in members_dict.keys()):
            return True

        return super().has_attribute(key)

    def get_attribute(self, key):
        attribute = super().get_attribute(key)
        if attribute is not None and attribute.has_value():
            return attribute

        members = inspect.getmembers(self.internal_module)
        members_dict = {}
        for member in members:
            members_dict[member[0]] = member[1]

        if not (key in members_dict.keys()):
            return attribute

        attr_v = members_dict[key]

        attribute.is_non_volatile = True
        v = parse_instance(self, key, attr_v, None)
        attribute.revise(v)

        return attribute

    def set_default_value(self, key, value):
        attribute = super().get_attribute(key)
        attribute.revise(value)

class AttributeHistory:
    def __init__(self, value : 'Value'):
        self.value = value

class Attribute:
    def __init__(self, name : 'str'):
        self.name = name
        self.history = []
        self.rev_history = {}
        self.access_num = 0
        self.rev_access_num = {}
        self.parent = None

        # a value which is contained in this attribute at first
        self.initial_value = None

        # if it is non-volatile, a value in this attribute is saved after running
        self.is_non_volatile = False

        register_field(self)

    def revise(self, value : 'Value'):
        # assgin name to the value
        if value.name == "":
            value.name = self.name

        if self.initial_value is None:
            self.initial_value = value

        hist = AttributeHistory(value)
        self.history.append(hist)

    def has_value(self):
        return len(self.history) > 0

    def get_value(self, inc_access = True):
        assert len(self.history) > 0
        if inc_access:
            self.access_num += 1
        return self.history[-1].value

    def commit(self, commit_id : 'str'):
        self.rev_history[commit_id] = self.history.copy()
        self.rev_access_num[commit_id] = self.access_num

    def checkout(self, commit_id : 'str'):
        if commit_id in self.rev_history:
            self.history = self.rev_history[commit_id].copy()
            self.access_num = self.rev_access_num[commit_id]
        else:
            self.history = []
            self.access_num = 0

    def has_diff(self, commit_id1 : 'str', commit_id2 : 'str'):
        if not commit_id1 in self.rev_history.keys() and not commit_id2 in self.rev_history.keys():
            return False

        if commit_id1 in self.rev_history.keys() and not commit_id2 in self.rev_history.keys():
            return True

        if not commit_id1 in self.rev_history.keys() and commit_id2 in self.rev_history.keys():
            return True

        if len(self.rev_history[commit_id1]) != len(self.rev_history[commit_id2]):
            return True
        for i in range(len(self.rev_history[commit_id1])):
            if self.rev_history[commit_id1][i] != self.rev_history[commit_id2][i]:
                return True

        return False

    def has_accessed(self, commit_id1 : 'str', commit_id2 : 'str'):
        if not commit_id1 in self.rev_access_num.keys() and not commit_id2 in self.rev_access_num.keys():
            return False

        if commit_id1 in self.rev_access_num.keys() and not commit_id2 in self.rev_access_num.keys():
            return False

        if not commit_id1 in self.rev_access_num.keys() and commit_id2 in self.rev_access_num.keys():
            return False

        return self.rev_access_num[commit_id1] != self.rev_access_num[commit_id2]

    def __str__(self):
        return self.name

class ValueHistory():
    def __init__(self, value, id):
        self.value = value
        self.id = id

class Value():
    def __init__(self):
        self.name = ""
        self.generator = None
        self.modifiers = []
        
        self.internal_value = None
        self.histories = {}
        self.history_id = utils.get_guid()
        self.id = utils.get_guid()
        register_value(self)

    def get_value(self) -> 'Value':
        return self

    def get_field(self) -> 'Field':
        return None

    def has_value(self) -> 'bool':
        return True

    def try_get_and_store_value(self, name : 'str') -> 'Value':
        return None

    def has_diff(self, commit_id1 : 'str', commit_id2 : 'str'):
        if not commit_id1 in self.histories and not commit_id2 in self.histories:
            return False
        if not commit_id1 in self.histories and commit_id2 in self.histories:
            return True
        if commit_id1 in self.histories and not commit_id2 in self.histories:
            return True
        return self.histories[commit_id1].id != self.histories[commit_id2].id

    def commit(self, commit_id : 'str'):
        self.histories[commit_id] = ValueHistory(self.internal_value, self.history_id)

    def checkout(self, commit_id : 'str'):
        if commit_id in self.histories:
            self.internal_value = self.histories[commit_id].value

    def modify(self, modifier, new_value):
        self.modifiers.append(modifier)
        self.history_id = utils.get_guid()

    def __str__(self):
        return self.name

class NoneValue(Value):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return self.name + '({})'.format('None')

class NumberValue(Value):
    def __init__(self, number):
        super().__init__()
        self.internal_value = number

    def __str__(self):
        if self.internal_value == None:
            return self.name + '(N.{})'.format('Any')
        return self.name + '(N.{})'.format(self.internal_value)

class StrValue(Value):
    def __init__(self, string):
        super().__init__()
        self.internal_value = string

    def __str__(self):
        if self.internal_value == None:
            return self.name + '(S.{})'.format('Any')
        return self.name + '(S.{})'.format(self.internal_value)

class BoolValue(Value):
    def __init__(self, b):
        super().__init__()
        self.internal_value = b

    def __str__(self):
        if self.internal_value == None:
            return self.name + '(B.{})'.format('Any')
        return self.name + '(B.{})'.format(self.internal_value)

class RangeValue(Value):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return self.name + '(R)'

class TupleValue(Value):
    def __init__(self, values = []):
        super().__init__()
        self.values = values
    def __str__(self):
        return self.name + '({})'.format(",".join([str(x) for x in self.values]))

class FuncValue(Value):
    def __init__(self, func : 'functions.FunctionBase', value : 'Value'):
        super().__init__()
        self.func = func
        self.value = value
    def __str__(self):
        return self.name + '(F)'

class ListValue(Value):
    def __init__(self, values = None):
        super().__init__()
        self.is_any = values is None
        self.attributes = Field()
        self.values = []
        self.append_func = FuncValue(functions_builtin.AppendFunction(self), self)
        self.attributes.get_attribute('append').revise(self.append_func)

    def get_field(self) -> 'Field':
        return self.attributes

    def __str__(self):
        return self.name + '(L)'

class DictValue(Value):
    def __init__(self):
        super().__init__()
        self.attributes = Field()

    def get_field(self) -> 'Field':
        return self.attributes

    def __str__(self):
        return self.name + '(D)'

class TensorValue(Value):
    def __init__(self):
        super().__init__()
        self.shape = ()
        self.value = None
    def __str__(self):
        return self.name + '(T.{})'.format(self.shape)

class Type(Value):
    def __init__(self, name : 'str'):
        super().__init__()
        self.name = name

class Instance(Value):
    def __init__(self, module : 'Field', inst, classinfo):
        super().__init__()
        self.attributes = Field()
        self.inst = inst
        self.callable = False
        self.func = None
        self.module = module
        self.classinfo = classinfo

    def get_field(self) -> 'Field':
        return self.attributes

class UserDefinedInstance(Instance):
    def __init__(self, module : 'Field', inst, classinfo, is_chainer_link = False):
        super().__init__(module, inst, classinfo)
        self.is_chainer_link = is_chainer_link

        if self.is_chainer_link:
            self.callable = True
            self.func = self.try_get_and_store_value('forward')

    def try_get_and_store_value(self, name : 'str') -> 'Value':

        attribute = self.attributes.get_attribute(name)
        if attribute.has_value():
            return attribute.get_value()

        if self.inst is not None:
            if not hasattr(self.inst, name):
                return None

            attr_v = getattr(self.inst, name)

            attribute.is_non_volatile = True
            v = parse_instance(self.attributes.module, name, attr_v, self)
            attribute.revise(v)

            return v
        else:
            members = inspect.getmembers(self.classinfo)
            members_dict = {}
            for member in members:
                members_dict[member[0]] = member[1]

            if not (name in members_dict.keys()):
                return None


            attribute.is_non_volatile = True
            v = parse_instance(self.attributes.module, name, members_dict[name], self)
            attribute.revise(v)

            return v
