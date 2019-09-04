import os
import numpy as np
from chainer_compiler.elichika.parser import config
from chainer_compiler.elichika.parser import values
import inspect
import re
import gast

current_id = 0

slice_int_max = 2 ** 31 - 1

dtype_float32 = np.array(1.0, dtype=np.float32).dtype
dtype_float64 = np.array(1.0, dtype=np.float64).dtype
dtype_int = np.array(1.0, dtype=np.int).dtype

def get_guid():
    global current_id
    id = current_id
    current_id += 1
    return id


def reset_guid():
    global current_id
    current_id = 0

def print_warning(s, lineprop):
    print('warning : {} in {}'.format(s, lineprop))

def print_error(s, lineprop):
    print('error : {} in {}'.format(s, lineprop))

def is_disabled_module(m):
    return m in config.disabled_modules

def str_2_dtype(str_dtype):
    if str_dtype == 'q':
        dtype = np.int64
    elif str_dtype == 'i':
        dtype = np.int32
    elif str_dtype == 'g':
        dtype = np.float64
    elif str_dtype == 'f':
        dtype = np.float32
    else:
        assert(False)
    return dtype

def create_obj_value_name_with_attribute(name: "str", pre_name: "str"):
    if len(pre_name) > 0 and pre_name[0] != '@':
        return pre_name
    else:
        return name

def lambda_source(l):
    s = inspect.getsource(l)
    if len(re.findall('lambda.*?:', s)) > 1:
        return None

    s = s[re.search('lambda.*?:', s).start():]
    min_length = len('lambda:_')  # shortest possible lambda expression
    while len(s) > min_length:
        try:
            code = compile(s, '<unused filename>', 'eval')
            return s.strip()
        except SyntaxError:
            s = s[:-1]
    return None

def clip_head(s: 'str'):
    splitted = s.split('\n')
    
    # remove comments
    comment_count = 0
    indent_targets = []
    for sp in splitted:
        if '"""' in sp or "'''" in sp:
            comment_count += 1
        else:
            if comment_count % 2 == 0:
                indent_targets.append(sp)

    hs = os.path.commonprefix(list(filter(lambda x: x != '', indent_targets)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    strs = map(lambda x: x[ls:], splitted)
    return '\n'.join(strs)

def try_get_obj(value, name, lineprop) -> 'values.Object':
    if value is None:
        print_warning('Failed to get value in "{}".'.format(name), lineprop)
        return None

    if isinstance(value, values.Value):
        assert(False)

    if isinstance(value, values.Attribute):
        if value.has_obj():
            return value.get_obj()

    if isinstance(value, values.Object):
        return value

    return None

def try_get_value(value, name, lineprop, is_none_allowed = False) -> 'values.Value':
    if value is None:
        print_warning('Failed to get value in "{}".'.format(name), lineprop)
        return None

    if isinstance(value, values.NoneValue) and not is_none_allowed:
        if config.show_warnings:
            print('Value {} is none. in {}'.format(name, lineprop))
        return None

    if isinstance(value, values.Value):
        return value

    if isinstance(value, values.Object):
        return value.get_value()

    if isinstance(value, values.Attribute):
        return value.get_obj().get_value()

    raise Exception('Value {} is invalid. in L.{}'.format(name, lineprop))

class LineProperty():
    def __init__(self, lineno=-1, filename=''):
        self.lineno = lineno
        self.filename = filename

    def get_line_str(self) -> 'str':
        return 'L.' + str(self.lineno)

    def __str__(self):

        if self.filename == '':
            return 'L.' + str(self.lineno)

        return self.filename + '[L.' + str(self.lineno) + ']'

class UnimplementedError(Exception):
    
    def __init__(self, message, lineprop):
        self.message = message
        self.lineprop = lineprop

    def __str__(self):
        return self.message + ' in ' + str(self.lineprop)

class DummyFlag:
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False


# ============================ Display utils ===================================

def intercalate(strings, sep):
    if strings == []:
        return ""
    return "".join([s + sep for s in strings[:-1]]) + strings[-1]


def expr_to_str(node):
    if isinstance(node, gast.BoolOp):
        return intercalate([expr_to_str(e) for e in node.values],
                " " + boolop_to_str(node.op) + " ")
    if isinstance(node, gast.BinOp):
        return "{} {} {}".format(expr_to_str(node.left),
                operator_to_str(node.op), expr_to_str(node.right))
    if isinstance(node, gast.UnaryOp):
        return "{}{}".format(unaryop_to_str(node.op), expr_to_str(node.operand))
    if isinstance(node, gast.Call):
        return "{}({})".format(expr_to_str(node.func),
                intercalate([expr_to_str(arg) for arg in node.args], ", "))
    if isinstance(node, gast.Num):
        return str(node.n)
    if isinstance(node, gast.Str):
        return "\"...\""  # sometimes it is too long
    if isinstance(node, gast.Attribute):
        return "{}.{}".format(expr_to_str(node.value), node.attr)
    if isinstance(node, gast.Subscript):
        return "{}[{}]".format(expr_to_str(node.value), slice_to_str(node.slice))
    if isinstance(node, gast.Name):
        return node.id
    if isinstance(node, gast.List):
        return "[" + intercalate([expr_to_str(e) for e in node.elts], ", ") + "]"
    if isinstance(node, gast.Tuple):
        return "(" + intercalate([expr_to_str(e) for e in node.elts], ", ") + ")"
    return ""


def boolop_to_str(node):
    if isinstance(node, gast.And):
        return "and"
    if isinstance(node, gast.Or):
        return "or"


def operator_to_str(node):
    if isinstance(node, gast.Add):
        return "+"
    if isinstance(node, gast.Sub):
        return "-"
    if isinstance(node, gast.Mult):
        return "*"
    if isinstance(node, gast.Div):
        return "/"
    if isinstance(node, gast.FloorDiv):
        return "//"


def unaryop_to_str(node):
    if isinstance(node, gast.Invert):
        return "!"  # 合ってる?
    if isinstance(node, gast.Not):
        return "not"
    if isinstance(node, gast.UAdd):
        return "+"
    if isinstance(node, gast.USub):
        return "-"


def slice_to_str(node):
    if isinstance(node, gast.Slice):
        ret = ""
        if node.lower: ret += expr_to_str(node.lower)
        if node.upper: ret += ":" + expr_to_str(node.upper)
        if node.step: ret += ":" + expr_to_str(node.step)
        if ret == "": ret = ":"
        return ret
    if isinstance(node, gast.ExtSlice):
        return intercalate([slice_to_str(s) for s in node.dims], ", ")
    if isinstance(node, gast.Index):
        return expr_to_str(node.value)


def is_expr(node):
    return isinstance(node, gast.BoolOp) \
            or isinstance(node, gast.BinOp) \
            or isinstance(node, gast.UnaryOp) \
            or isinstance(node, gast.Lambda) \
            or isinstance(node, gast.IfExp) \
            or isinstance(node, gast.Dict) \
            or isinstance(node, gast.Set) \
            or isinstance(node, gast.ListComp) \
            or isinstance(node, gast.SetComp) \
            or isinstance(node, gast.DictComp) \
            or isinstance(node, gast.GeneratorExp) \
            or isinstance(node, gast.Await) \
            or isinstance(node, gast.Yield) \
            or isinstance(node, gast.YieldFrom) \
            or isinstance(node, gast.Compare) \
            or isinstance(node, gast.Call) \
            or isinstance(node, gast.Repr) \
            or isinstance(node, gast.Num) \
            or isinstance(node, gast.Str) \
            or isinstance(node, gast.FormattedValue) \
            or isinstance(node, gast.JoinedStr) \
            or isinstance(node, gast.Bytes) \
            or isinstance(node, gast.NameConstant) \
            or isinstance(node, gast.Ellipsis) \
            or isinstance(node, gast.Attribute) \
            or isinstance(node, gast.Subscript) \
            or isinstance(node, gast.Starred) \
            or isinstance(node, gast.Name) \
            or isinstance(node, gast.List) \
            or isinstance(node, gast.Tuple)


def node_description(node):
    type_name = type(node).__name__
    lineno = " (line {})".format(node.lineno) if hasattr(node, 'lineno') else ""
    if isinstance(node, gast.FunctionDef):
        return "{} {}{}".format(type_name, node.name, lineno)
    if is_expr(node):
        return "{} {}{}".format(type_name, expr_to_str(node), lineno)
    return type_name
