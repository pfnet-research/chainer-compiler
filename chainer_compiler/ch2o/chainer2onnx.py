# coding: utf-8

import ast
import gast
import inspect

import numpy as np
import onnx
from onnx import checker
from onnx import helper
from onnx import TensorProto

import code
import logging
import sys
import types

import chainer
import numpy

from chainer_compiler.ch2o.test_args import dprint
from chainer_compiler.ch2o.env import Env
from chainer_compiler.ch2o.utils import new_tensor, new_sequence, clip_head, ValueReturn, istensor, totensor, make_graph
from chainer_compiler.ch2o.links import Link2NodeClass
from chainer_compiler.ch2o.funcs import Func, Func2NodeClass, Function_Concat, Function_Dummy, castto
from chainer_compiler.ch2o.builtin_funcs import builtin_functions
from chainer_compiler.ch2o.value import Value

import builtins


id2name_list = []


def init_id2name(ch):
    global id2name_list
    id2name_list = []
    for k, v in ch.namedlinks():
        # print('add link',k,v,id(v))
        id2name_list.append((id(v), k))


def id2name(nid):
    # print('nid',nid)
    for k, v in id2name_list:
        if k == nid:
            return v
    raise Exception("Not Found ID ", nid)


def _value(v):
    if (isinstance(v, User_Defined_Function) or
        isinstance(v, User_Defined_Func_In_Link)):
        return v
    return Value(v)


def convert_link(ch, env):
    res = None
    if type(ch) in Link2NodeClass:
        res = Link2NodeClass[type(ch)](ch)
    else:
        res = User_Defined_Link(ch, env)

    ts = res.init_tensors()
    if len(ts) != 0:
        pathname = id2name(id(ch))
        env.add_init(ts, pathname)
    return res


class Function_base(object):
    def stub_call(self, args, kwargs, loenv):
        # 関数引数は inspect.signature できれいにしたい

        astargs = list(map(lambda x: x.id, self.ast.args.args))
        args = dict(zip(astargs, args))

        defs = self.ast.args.defaults
        d = len(astargs) - len(args.keys())
        if d > 0:
            for i, v in enumerate(defs[::-1][:d]):
                args.update({astargs[-i-1]: eval_ast(v, loenv)})

        args.update(kwargs)

        assert(len(astargs) == len(args.keys()))
        loenv.update_vars(args)

        # このやり方は、If文などでコントロールフローが別れるような場合に
        # 複数ヶ所の return を変換する際に問題になる
        try:
            eval_ast(self.ast.body, loenv)
            return None
        except ValueReturn as v:
            return v.value


class User_Defined_Function(Function_base):
    def __init__(self, func):
        self.func = func
        src = clip_head(inspect.getsource(func))
        dprint(src)
        self.ast = gast.ast_to_gast(ast.parse(src)).body[0]
        assert(isinstance(self.ast, gast.gast.FunctionDef))

    def call(self, args, kwargs, env):
        loenv = env.localenv(sys.modules[self.func.__module__])
        return self.stub_call(args, kwargs, loenv)


class User_Defined_Func_In_Link(Function_base):
    def __init__(self, ch, fn):
        self.ch = ch
        src = clip_head(inspect.getsource(fn))
        dprint(src)
        self.ast = gast.ast_to_gast(ast.parse(src)).body[0]
        assert(isinstance(self.ast, gast.gast.FunctionDef))

    def call(self, args, kwargs, env):
        loenv = env.localenv(sys.modules[self.ch.__module__])
        args = [self.ch] + args
        return self.stub_call(args, kwargs, loenv)


class User_Defined_Link(object):
    def __init__(self, ch, env):
        src = clip_head(inspect.getsource(ch.forward))
        dprint(src)
        self.ast = gast.ast_to_gast(ast.parse(src)).body[0]

        self.call = User_Defined_Func_In_Link(ch, ch.forward).call

        # 以下、 最初の外からのためのやつ
        # code.InteractiveConsole({'v': self.ast}).interact()
        self.forward_arglen = len(self.ast.args.args)-1

        # ここで、初期化したやつを上書きしてやる必要が出てくる
        # あとでchainerで実行するために回復しないといけないので、
        # restore_funcs に復元すべきものを追加している
        self.inits = []

        for s, v in ch.namedparams():
            s = s[1:]
            if s.find('/') != -1:
                continue
            t = helper.make_tensor_value_info(
                '/'+s, TensorProto.FLOAT, list(v.shape))
            self.inits.append(t)
            mv = getattr(ch, s)
            setattr(ch, s, t)
            env.restore_funcs.append(lambda: setattr(ch, s, mv))

        # TODO(satos) Yieldをコンパイルできるとこれを消せる
        mv = getattr(ch, 'children')
        setattr(ch, 'children', Func(lambda _, __, ___: mv()))
        env.restore_funcs.append(lambda: setattr(ch, 'children', mv))

    def init_tensors(self):
        return self.inits


class User_Defined_Class(object):
    def __init__(self, classtype):
        # classtypeのmethod は持ってるが init は呼ばれてない、というobjectが必要になる。
        # ので、あえて parent のinit を呼ばない継承をする
        class Tmp(classtype):
            def __init__(_):
                pass

        # dprint('user defined class of',classtype)
        ch = Tmp()
        ch.__module__ = classtype.__module__

        # code.InteractiveConsole({'Tmp': Tmp,'v': ch}).interact()
        def f(args, kwargs, env):
            if not isinstance(classtype.__init__, type(str.__init__)):  # slot wrapper というものらしい
                User_Defined_Func_In_Link(
                    ch, classtype.__init__).call(args, kwargs, env)

            return ch

        self.init_wrapper = Func(f)


import logging

# logging. なんとか
# print( なんとか )
# はデバッグ出力なのでコンパイルせずに読み飛ばしたい


def is_print_logging(s, env):
    return (
        isinstance(s, gast.Expr) and
        isinstance(s.value, gast.Call) and
        isinstance(s.value.func, gast.Attribute) and
        isinstance(eval_ast(s.value.func.value, env), logging.__class__)
    ) or (
        isinstance(s, gast.Expr) and
        isinstance(s.value, gast.Call) and
        isinstance(s.value.func, gast.Name) and
        s.value.func.id == 'print'
    )


def _find_in_out(localenv, env):
    used_onnx_names = set()
    for node in localenv.nodes:
        used_onnx_names |= set(node.input)

    outer_vars = env.get_var_dict()
    inner_vars = localenv.get_var_dict()

    # A tuple of (in-value, out-value, extra info for later setattr)
    # keyed by a variable name.
    in_out = {}
    for key, iv in inner_vars.items():
        ov = outer_vars.get(key, None)
        if isinstance(ov, Value):
            # Changing link or something to Value is not supported.
            assert isinstance(iv, Value), '%s => %s' % (ov, iv)
        elif ov is None or iv is None:
            pass
        else:
            # Changing Value to link or something is not supported.
            assert not isinstance(iv, Value), '%s => %s' % (ov, iv)
            continue

        if ov is None or iv is None or ov.value != iv.value:
            in_out[key] = (ov, iv, None)
            continue

        if ov.to_value_info(env).name in used_onnx_names:
            in_out[key] = (ov, None, None)

    var_ids = {}
    def attr_id(var, key):
        vid = id(var.value)
        if vid not in var_ids:
            var_ids[vid] = 'v%d' % (len(var_ids) + 1)
        return var_ids[vid] + '.' + key

    in_attrs = {}
    for var, key, value in localenv.read_attrs:
        k = attr_id(var, key)
        if k not in in_attrs:
            in_attrs[k] = value

    out_attrs = {}
    for var, key, value in localenv.wrote_attrs:
        k = attr_id(var, key)
        out_attrs[k] = (value, (var, key))

    for k in set(list(in_attrs.keys()) + list(out_attrs.keys())):
        iv = in_attrs.get(k, None)
        ov, setattr_info = out_attrs.get(k, (None, None))
        in_out[k] = (iv, ov, setattr_info)

    # ループ内で使われた link パラメータは
    # 1. 外の env にコピーしなければならない
    env.init_tensors.update(localenv.init_tensors)
    # 2. state としてループ内に持ち込まなければならない
    for init in localenv.init_tensors.values():
        key = '/' + init.name
        in_out[key] = (Value(init), None, None)

    return in_out


def eval_if(nast, env):
    cond = eval_ast(nast.test, env)
    if cond.is_py and cond.value is True:
        return eval_ast(nast.body, env)
    elif cond.is_py and cond.value is False:
        return eval_ast(nast.orelse, env)

    then_env = env.new_block()
    ty = eval_ast(nast.body, then_env)
    assert ty.is_none()

    else_env = env.new_block()
    ty = eval_ast(nast.orelse, else_env)
    assert ty.is_none()

    then_in_out = _find_in_out(then_env, env)
    else_in_out = _find_in_out(else_env, env)
    keys = set(list(then_in_out.keys()) + list(else_in_out.keys()))

    input_values = []
    then_outputs = []
    else_outputs = []
    final_outputs = []
    final_setattrs = []

    for key in keys:
        then_iv, then_ov, then_setattr_info = then_in_out.get(
            key, (None, None, None))
        else_iv, else_ov, else_setattr_info = else_in_out.get(
            key, (None, None, None))

        if then_setattr_info is None:
            setattr_info = else_setattr_info
        else:
            if else_setattr_info is not None:
                assert then_setattr_info == else_setattr_info
            setattr_info = then_setattr_info

        def set_final_output(key, out):
            out = out.copy(env, name=key)
            final_outputs.append((key, out.value))
            if setattr_info is not None:
                final_setattrs.append(tuple(list(setattr_info) + [out]))

        iv = else_iv if then_iv is None else then_iv
        if iv is None:
            iv = Value(False)
        input_values.append(iv.to_value_info(env))

        if then_ov is None and else_ov is None:
            continue
        if then_ov is None:
            then_outputs.append(iv.to_value_info(env))
            else_outputs.append(else_ov.to_value_info(else_env))
            set_final_output(key, else_ov)
        elif else_ov is None:
            then_outputs.append(then_ov.to_value_info(then_env))
            else_outputs.append(iv.to_value_info(env))
            set_final_output(key, then_ov)
        else:
            then_outputs.append(then_ov.to_value_info(then_env))
            else_outputs.append(else_ov.to_value_info(else_env))
            set_final_output(key, then_ov)

    then_graph = make_graph(
        then_env.nodes,
        "If_then",
        input_values,
        then_outputs,
    )

    else_graph = make_graph(
        else_env.nodes,
        "If_else",
        input_values,
        else_outputs,
    )

    env.addnode(
        'If',
        inputs=[cond.to_value_info(env).name] + [i.name for i in input_values],
        outputs=[o.name for _, o in final_outputs],
        then_branch=then_graph,
        else_branch=else_graph,
    )

    for k, o in final_outputs:
        env.set_var(k, _value(o))

    for var, key, value in final_setattrs:
        setattr(var.value, key, value)

    return None


def eval_for(nast, env):
    assert nast.orelse == []
    ite = eval_ast(nast.iter, env)

    # A hack for ResNet50.
    # TODO(hamaji): Come up with a sophisticated way.
    # TODO(hamaji): This code doesn't handle scope properly, I think.
    if (isinstance(ite.value, types.GeneratorType) and
        'ChainList.children' in str(ite.value)):
        # とりあえず実際にfor文を回す
        tg = nast.target.id
        env.set_var(tg, Value(None))
        for v in ite.value:
            env.set_var(tg, _value(v))
            eval_ast(nast.body, env)
            # print('looping',env.vars.keys())

        env.pop_var(tg)
        return None

    if ite.is_py:
        ite = Value([Value(v) for v in ite.value])

    assert isinstance(nast.target, gast.Name)
    x = nast.target.id

    # 新たなenv を作って、評価中にできた子グラフをもとにする
    localenv = env.new_block()

    cnt = new_tensor()
    gtx = new_sequence()
    localenv.set_var(x, _value(localenv.calc(
        "SequenceAt",
        inputs=[gtx.name, cnt.name],
    )))
    ty = eval_ast(nast.body, localenv)
    assert ty.is_none()

    in_out = _find_in_out(localenv, env)

    input_values = []
    output_values = []
    final_outputs = []
    final_setattrs = []
    for key, (iv, ov, setattr_info) in in_out.items():
        if ov is None:
            continue
        if iv is None:
            iv = Value(False)
        out = ov.copy(env, name=key)
        final_outputs.append((key, out.value))
        if setattr_info is not None:
            final_setattrs.append(tuple(list(setattr_info) + [out]))
        input_values.append(iv.to_value_info(env))
        output_values.append(ov.to_value_info(env))

    cond = new_tensor(name='loop_cond')
    localgraph = make_graph(
        localenv.nodes,
        "Loop_subgraph",
        [cnt, cond, gtx] + input_values,
        [cond, gtx] + output_values
    )

    mtc = env.calc(
        "ChainerGenericLen",
        inputs=[ite.to_sequence(env).name],
    )

    env.addnode(
        'Loop',
        inputs=([mtc.name, "", ite.to_sequence(env).name] +
                [i.name for i in input_values]),
        outputs=([new_tensor('out_generator').name] +
                 [o.name for _, o in final_outputs]),
        body=localgraph
    )

    for k, o in final_outputs:
        if '.' not in k and '/' not in k:
            env.set_var(k, _value(o))

    for var, key, value in final_setattrs:
        setattr(var.value, key, value)

    return None


def eval_assign(nast, env):
    value = eval_ast(nast.value, env)
    targs = nast.targets
    assert(len(targs) == 1)
    # v,w = 1 も targetsは長さ1のlistになるので len(rargs) != 1 の状況は謎ですね

    # tgとして、下以外に
    # List, ListのIndex, Starred
    # またこれらを再帰的に組み合わせたものが存在しうる

    def set_var(k, v):
        v = _value(v)
        if not v.is_py:
            v = v.identity(env, name=k)
        env.set_var(k, v)

    tg = targs[0]
    if isinstance(tg, gast.Name):
        set_var(tg.id, value)
    elif isinstance(tg, gast.Tuple):
        assert(isinstance(value.value, tuple))
        value = value.value
        assert(len(tg.elts) == len(value))

        for i, v in enumerate(value):
            set_var(tg.elts[i].id, v)  # TODO(satos) これこのあと更に再帰的に書く必要あるかも

    elif isinstance(tg, gast.Attribute):
        body = eval_ast(tg.value, env)
        # If the attr already exists, de-literalize and push it to
        # `read_attrs` by calling `get_attribute`. See lazy_self_init
        # test in ForAndIf.py.
        if hasattr(body.value, tg.attr):
            body.get_attribute(tg.attr, env)
        env.wrote_attrs.append((body, tg.attr, value))
        setattr(body.value, tg.attr, value)
    else:
        raise Exception('invalid assing lvalue', targs[0])
    return None


def eval_call(nast, env):
    fn = eval_ast(nast.func, env)
    if not fn.is_py:
        raise TypeError('Expected a callable: %s' % fn.value)
    fn = fn.value

    # TODO(hamaji): Merge this logic with is_print_logging. Also,
    # maybe it's better to try emitting ChainerPrint.
    if fn in (logging.debug, logging.info,
              logging.warn, logging.warning, logging.error):
        return None

    args = []
    for ag in nast.args:
        if isinstance(ag, gast.Starred):
            args += list(eval_ast(ag.value, env))
        else:
            args.append(eval_ast(ag, env))

    keywords = dict(
        map(lambda x: (x.arg, eval_ast(x.value, env)), nast.keywords))

    # code.InteractiveConsole({'fn': fn}).interact()

    # chainer.functions の関数とかは、ここでfookをかける。
    if fn in Func2NodeClass.keys():
        return Func2NodeClass[fn].call(args, keywords, env)

    dprint(fn, fn.__class__)
    if isinstance(fn, types.FunctionType):
        fn = User_Defined_Function(fn)
    elif isinstance(fn, types.MethodType):
        # apply はforwardにする
        # code.InteractiveConsole({'fn': fn}).interact()
        if fn.__func__ == chainer.FunctionNode.apply:
            fn = User_Defined_Func_In_Link(
                fn.__self__, fn.__self__.forward)
        elif fn.__func__ == chainer.FunctionNode.retain_inputs:
            # TODO(satos) これbackward側に何か伝える必要がありそう
            fn = Func(lambda _, __, ___: None)
        else:
            fn = User_Defined_Func_In_Link(fn.__self__, fn)
    elif fn in builtin_functions:
        fn = builtin_functions[fn]
    elif isinstance(fn, type):
        # なにがしかのinstanceを作成したはず
        assert fn.__module__ != 'builtins'
        fn = User_Defined_Class(fn).init_wrapper
    elif isinstance(fn, chainer.link.Link):
        fn = convert_link(fn, env)

    dprint('converted to', fn)
    return fn.call(args, keywords, env)


def eval_unary_op(nast, env):
    v = eval_ast(nast.operand, env)
    res = new_tensor()
    if isinstance(nast.op, gast.USub):
        # optype = "*= -1"
        def opfun(x): return -x
    elif isinstance(nast.op, gast.Not):
        # optype = "Not"
        def opfun(x): return not x
    else:
        raise Exception('unknown operator', nast.op)

    if not istensor(v):
        return opfun(v.value)
    else:
        raise Exception("Unimplemented yet")


def eval_binary_op(nast, env):
    lv = eval_ast(nast.left, env)
    rv = eval_ast(nast.right, env)

    res = new_tensor(['TODO'])
    isfloor = False
    if isinstance(nast.op, gast.Add):
        optype = "Add"

        def opfun(a, b): return a + b

    elif isinstance(nast.op, gast.Sub):
        optype = "Sub"

        def opfun(a, b): return a - b

    elif isinstance(nast.op, gast.Mult):
        optype = "Mul"

        def opfun(a, b): return a * b

    elif isinstance(nast.op, gast.FloorDiv):
        optype = "Div"
        isfloor = True

        def opfun(a, b): return a // b

    elif isinstance(nast.op, gast.Div):
        optype = "Div"

        def opfun(a, b): return a / b

    else:
        raise Exception('unknown operator', nast.op)

    # code.InteractiveConsole({'lv': lv, 'rv': rv}).interact()

    # TODO(hamaji): Reconsider if constant folding is necessary in CH2O.
    #if not istensor(lv) and not istensor(rv):
    #    # 定数畳み込みを行う
    #    return opfun(lv, rv)

    lv.to_value_info(env)
    rv.to_value_info(env)
    if lv.is_sequence() and rv.is_sequence():
        assert optype == 'Add'
        lv = lv.to_sequence(env)
        rv = rv.to_sequence(env)

        state = new_sequence(name='seq_plus_state')
        cond = new_tensor(name='seq_plus_cond')
        index = new_tensor(name='seq_plus_index')
        elem = new_tensor(name='seq_plus_elem')
        out_state = new_tensor(name='seq_plus_out_state')
        nodes = []
        nodes.append(helper.make_node(
            'SequenceAt',
            inputs=[rv.name, index.name],
            outputs=[elem.name]
        ))
        nodes.append(helper.make_node(
            'ChainerSequenceAppend',
            inputs=[state.name, elem.name],
            outputs=[out_state.name]
        ))
        loop = make_graph(
            nodes,
            "SeqPlus",
            [index, cond, state],
            [cond, out_state],
        )

        length = env.calc('ChainerGenericLen', inputs=[rv.name])
        res = new_sequence(name='seq_plus')
        env.addnode(
            'Loop',
            inputs=[length.name, "", lv.name],
            outputs=[res.name],
            body=loop
        )
    else:
        if optype == 'Div' and not isfloor:
            lv = castto(lv.to_tensor(env), TensorProto.FLOAT, env)
            rv = castto(rv.to_tensor(env), TensorProto.FLOAT, env)
        else:
            lv = lv.to_tensor(env)
            rv = rv.to_tensor(env)
        res = env.calc(
            optype,
            inputs=[lv.name, rv.name],
        )

    if isfloor:
        res = env.calc(
            "Floor",
            inputs=[res.name],
        )

    return res


def eval_attribute(nast, env):
    body = eval_ast(nast.value, env)

    if not body.is_py:
        if nast.attr == 'shape':
            res = env.calc(
                'Shape',
                inputs=[body.to_tensor(env).name],
                npdtype=np.int64,
            )
            res = env.calc_seq(
                'ChainerSequenceSeparate',
                inputs=[res.name],
            )
            return res

        elif nast.attr == 'size':
            res = env.calc(
                'Size',
                inputs=[body.to_tensor(env).name],
                npdtype=np.int64,
            )
            return res

        elif nast.attr == 'append':
            # TODO(satos) ごまかさない
            assert isinstance(
                nast.value, gast.Name) and nast.value.id in env.get_var_dict().keys()
            na = nast.value.id

            # あと、ここのnaがreferenceの場合不正確
            # たとえば
            # x = y
            # x.append(3)
            # のyが更新されないので問題

            def f(args, _, env):
                assert len(args) == 1
                v = args[0].to_tensor(env)
                env.set_var(na, _value(env.calc_seq(
                    'ChainerSequenceAppend',
                    inputs=[body.to_sequence(env).name, v.name],
                )))
                return None

            return Func(f)

        raise Exception('Unimplemented attribute ',
                        nast.attr, ' for tensor')
    return body.get_attribute(nast.attr, env)


def eval_compare(nast, env):
    lv = eval_ast(nast.left, env)
    vs = [eval_ast(x, env) for x in nast.comparators]

    if env.outer_block is None and all(v.is_py for v in [lv] + vs):
        # Constant folding.
        lv = lv.value
        res = True
        for op, r in zip(nast.ops, vs):
            r = r.value
            if isinstance(op, gast.Eq):
                res = res and (lv == r)
            elif isinstance(op, gast.NotEq):
                res = res and (lv != r)
            elif isinstance(op, gast.Is):
                res = res and (lv is r)
            elif isinstance(op, gast.IsNot):
                res = res and (lv is not r)
            elif isinstance(op, gast.Gt):
                res = res and (lv > r)
            elif isinstance(op, gast.GtE):
                res = res and (lv >= r)
            elif isinstance(op, gast.Lt):
                res = res and (lv < r)
            elif isinstance(op, gast.LtE):
                res = res and (lv <= r)
            else:
                raise Exception('unimplemented operator', op)
        return res

    assert len(vs) == 1, 'Multiple comparator not implemented yet'
    res = None
    for op, r in zip(nast.ops, vs):
        needs_not = False
        if isinstance(op, gast.Eq):
            optype = 'Equal'
        elif isinstance(op, gast.NotEq):
            needs_not = True
            optype = 'Equal'
        elif isinstance(op, gast.Is):
            optype = 'ChainerGenericIs'
        elif isinstance(op, gast.IsNot):
            needs_not = True
            optype = 'ChainerGenericIs'
        elif isinstance(op, gast.Gt):
            optype = 'Greater'
        elif isinstance(op, gast.GtE):
            # TODO(hamaji): This computation is wrong for NaNs.
            needs_not = True
            optype = 'Less'
        elif isinstance(op, gast.Lt):
            optype = 'Less'
        elif isinstance(op, gast.LtE):
            # TODO(hamaji): This computation is wrong for NaNs.
            needs_not = True
            optype = 'Greater'
        else:
            raise Exception('unimplemented operator', op)

        res = env.calc(optype,
                       npdtype=np.bool,
                       inputs=[lv.to_value_info(env).name,
                               r.to_value_info(env).name])
        if needs_not:
            res = env.calc('Not', npdtype=np.bool, inputs=[res.name])
    return res



def eval_list_comp(nast, env):
    vn = "dummy@" + new_tensor().name  # 重ならない名前にする(ループ内ループもあるため)
    assert len(nast.generators) >= 1
    tast = gast.ast_to_gast(ast.parse("v.append(w)")).body[0]
    tast.value.func.value.id = vn
    tast.value.args[0] = nast.elt

    for gen in nast.generators:
        # とりあえず、このあたりはまだ実装しません
        assert len(gen.ifs) == 0 and gen.is_async == 0
        tast = gast.For(target=gen.target, iter=gen.iter,
                        body=[tast], orelse=[], type_comment=None)

    init = gast.ast_to_gast(ast.parse("v = []")).body[0]
    init.targets[0].id = vn
    tast = [init, tast]

    rv = eval_ast(tast, env)
    assert rv.is_none()
    res = env.pop_var(vn)
    return res


def _concat(xs, axis, env):
    assert isinstance(xs, tuple)  # 今のところ tuple 以外は concat できない
    return env.calc(
        "Concat",
        inputs=list(map(lambda x: x.name, xs)),
        axis=axis,
    )


def eval_subscript(nast, env):
    vs = eval_ast(nast.value, env)

    # TODO(hamaji): Use 2**63-1 instead.
    int_max = 2 ** 31 - 1

    def eval_with_default(nast, default_value):
        if nast is None:
            return Value(np.array(default_value)).to_tensor(env)
        return eval_ast(nast, env).to_tensor(env)

    if isinstance(nast.slice, gast.Index):
        if isinstance(nast.slice.value, gast.Tuple):
            assert vs.is_tensor(), 'Advanced indexing for Python list'
            indices = []
            slice_specs = []
            for index in nast.slice.value.elts:
                indices.append(eval_ast(index, env).to_tensor(env).name)
                slice_specs.append(1)
            return env.calc(
                'ChainerGetItem',
                inputs=[vs.to_tensor(env).name] + indices,
                slice_specs=slice_specs
            )

        index = eval_ast(nast.slice.value, env).to_tensor(env)
        if vs.is_sequence():
            return env.calc(
                'SequenceAt',
                inputs=[vs.to_sequence(env).name, index.name]
            )
        else:
            return env.calc(
                'ChainerGetItem',
                inputs=[vs.to_tensor(env).name, index.name],
                slice_specs=[1]
            )

    def get_slice_indices(slice):
        if slice.lower is None and slice.upper is None and slice.step is None:
            return []
        indices = [eval_with_default(slice.lower, 0).name,
                   eval_with_default(slice.upper, int_max).name]
        if slice.step is not None:
            indices.append(eval_with_default(slice.step, 1).name)
        return indices

    if isinstance(nast.slice, gast.Slice):
        indices = get_slice_indices(nast.slice)
        if vs.is_sequence():
            return env.calc_seq(
                'ChainerSequenceGetSlice',
                inputs=[vs.to_sequence(env).name] + indices
            )
        else:
            return env.calc(
                'ChainerGetItem',
                inputs=[vs.to_tensor(env).name] + indices,
                slice_specs=[len(indices)]
            )

    if isinstance(nast.slice, gast.ExtSlice):
        assert vs.is_tensor(), 'Advanced indexing for Python list'
        indices = []
        slice_specs = []
        for dim in nast.slice.dims:
            if isinstance(dim, gast.Index):
                indices.append(eval_ast(dim.value, env).to_tensor(env).name)
                slice_specs.append(1)
            elif isinstance(dim, gast.Slice):
                ni = get_slice_indices(dim)
                indices.extend(ni)
                slice_specs.append(len(ni))
            else:
                assert False, 'Unknown slice: %s in %s' % (dim, nast.slice)

        return env.calc(
            'ChainerGetItem',
            inputs=[vs.to_tensor(env).name] + indices,
            slice_specs=slice_specs
        )

    assert False, 'Unknown slice: %s' % nast.slice


def eval_list(nast, env):
    # Sequenceにしているが、ここはPythonのlistのままにしておきたいとのこと
    # Sequenceにする
    vs = list(map(lambda x: eval_ast(x, env), nast.elts))
    res = env.calc_seq(
        "ChainerSequenceCreate",
        inputs=[],
    )
    for v in vs:
        v = v.to_tensor(env)
        res = env.calc_seq(
            "ChainerSequenceAppend",
            inputs=[res.name, v.name],
        )
    return res


_eval_ast_depth = 0


def eval_ast(nast, env):
    for k, v in env.get_var_dict().items():
        assert not isinstance(v, onnx.ValueInfoProto), '%s %s' % (k, v)

    global _eval_ast_depth
    if not isinstance(nast, list):
        dprint('-' * _eval_ast_depth, gast.dump(nast), env.get_var_dict().keys())

    _eval_ast_depth += 1
    r = eval_ast_impl(nast, env)
    _eval_ast_depth -= 1
    return _value(r)


def eval_ast_impl(nast, env):
    if isinstance(nast, list):
        # 逐次実行
        for s in nast:
            if is_print_logging(s, env):
                continue
            eval_ast(s, env)
        return None
    elif isinstance(nast, gast.For):
        return eval_for(nast, env)

    elif isinstance(nast, gast.Assign):
        return eval_assign(nast, env)

    elif isinstance(nast, gast.AugAssign):
        # referenceへの代入に対してこれは不正確
        ca = gast.Assign(targets=[nast.target], value=gast.BinOp(
            left=nast.target, op=nast.op, right=nast.value))
        return eval_ast(ca, env)

    elif isinstance(nast, gast.Call):
        return eval_call(nast, env)

    elif isinstance(nast, gast.UnaryOp):
        return eval_unary_op(nast, env)

    elif isinstance(nast, gast.BinOp):
        return eval_binary_op(nast, env)

    elif isinstance(nast, gast.BoolOp):
        # 現在は定数boleanのみ対応
        vs = list(map(lambda x: eval_ast(x, env), nast.values))
        res = new_tensor()
        if isinstance(nast.op, gast.And):
            def opfun(v): return all(v)
        else:
            raise Exception('unknown operator', nast.op)

        if not any(map(istensor, vs)):
            return opfun(vs)

        raise Exception('Unimplemented BoolOp for tensor', nast)

    elif isinstance(nast, gast.Attribute):
        return eval_attribute(nast, env)

    elif isinstance(nast, gast.Compare):
        return eval_compare(nast, env)

    elif isinstance(nast, gast.If):
        return eval_if(nast, env)

    elif isinstance(nast, gast.ListComp):
        return eval_list_comp(nast, env)

    elif isinstance(nast, gast.Subscript):
        return eval_subscript(nast, env)

    elif isinstance(nast, gast.Delete):
        # おのおの単に忘れる
        vs = nast.targets
        for v in vs:
            assert isinstance(v, gast.Name)
            env.pop_var(v.id)
        return None

    elif isinstance(nast, gast.Name):
        try:
            return env.get_var(nast.id)
        except NameError as ne:
            if nast.id in dir(env.module):
                return getattr(env.module, nast.id)
            elif nast.id in dir(builtins):
                return getattr(builtins, nast.id)
            raise
    elif isinstance(nast, gast.Constant):
        return nast.value
    elif isinstance(nast, gast.Expr):
        return eval_ast(nast.value, env)
    elif isinstance(nast, gast.Constant) and isinstance(nast.value, str):
        return nast.value
    elif isinstance(nast, gast.Tuple):
        return tuple(map(lambda x: eval_ast(x, env), nast.elts))
    elif isinstance(nast, gast.List):
        return eval_list(nast, env)

    elif isinstance(nast, gast.Return):
        raise ValueReturn(eval_ast(nast.value, env))

    elif isinstance(nast, gast.Assert):
        # TODO(hamaji): Emit an assertion?
        return None

    # TODO(hamaji): Implement `with`.
    # elif isinstance(nast, gast.With):
    #     sys.stderr.write(
    #         'WARNING: Currenctly, the context of `with` is just ignored\n')
    #     for s in nast.body:
    #         eval_ast(s, env)
    #     return None

    else:
        print('unknown ast')
        code.InteractiveConsole({'nast': nast, 'env': env}).interact()
        raise Exception('unknown ast', nast)

    raise Exception("shouldn't reach here", nast)


def compile_model(model, inputs):
    # return helper.make_graph([],'dummy',[],[])

    init_id2name(model)
    # code.InteractiveConsole({'mo': model}).interact()
    env = Env(sys.modules[model.__module__])
    molk = User_Defined_Link(model, env)

    input_tensors = []
    for i in inputs:
        # TODO(hamaji): Set valid type info.
        if isinstance(i, (list, tuple)):
            x = new_sequence()
        elif i is None:
            x = new_tensor()
        else:
            if isinstance(i, int):
                i = np.array(i)
            else:
                # TODO(durswd): This code requires chainer6.x
                i = chainer.cuda.to_cpu(i)

            x = new_tensor(dims=i.shape, dtype=i.dtype)
        input_tensors.append(x)

    input_values = [Value(i) for i in input_tensors]
    v = molk.call(input_values, [], env)

    dprint('output_tensors', v)
    if isinstance(v.value, tuple):
        output_tensors = list(v.value)  # ばらしてみる
    else:
        output_tensors = [v]  # とりあえず1tensor

    # print('env.init_tensors ',env.init_tensors)
    input_tensors += list(env.init_tensors.values())

    for f in env.restore_funcs:
        f()

    # for no in env.nodes:
    #   print(no.op_type)
    # print(env.nodes)
    # print(input_tensors)
    # print(output_tensors)
    # for ch in model.namedparams():
    #    print(ch)

    outputs_vi = [o.to_value_info(env) for o in output_tensors]
    graph = make_graph(env.nodes,
                       'name_is_unknown_now',
                       input_tensors,
                       outputs_vi)

    # inputのうち、重みであるものにはinitializerをつける
    # batch_sizeやinput_sizeなどの可変なものはできる限りのそのままで

    # Chainer compiler 独自のノードを使うとcheckできなくなる...
    # checker.check_graph(graph)
    mo = helper.make_model(graph)

    # print(mo)
    return mo
