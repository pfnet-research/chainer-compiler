import ast
import gast

# ============================== Display utils =================================

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
        return "{}({}{})".format(expr_to_str(node.func),
                intercalate([expr_to_str(arg) for arg in node.args], ", "),
                intercalate(["{}={}".format(kwarg.arg, expr_to_str(kwarg.value))
                    for kwarg in node.keywords], ", "))
    if isinstance(node, gast.Num):
        return str(node.n)
    if isinstance(node, gast.Str):
        if len(node.s) < 20:
            return "\'" + node.s + "\'"
        return "\"...\""  # sometimes it is too long
    if isinstance(node, gast.Attribute):
        return "{}.{}".format(expr_to_str(node.value), node.attr)
    if isinstance(node, gast.Subscript):
        return "{}[{}]".format(expr_to_str(node.value), slice_to_str(node.slice))
    if isinstance(node, gast.NameConstant):
        return str(node.value)
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
        ret += ":"
        if node.upper: ret += expr_to_str(node.upper)
        ret += ":"
        if node.step: ret += expr_to_str(node.step)
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

# ==============================================================================

def add_dict(dest, src):
    for k, v in src.items():
        dest[k] = v

def find(seq, pred):
    for elt in seq:
        if pred(elt):
            return elt
