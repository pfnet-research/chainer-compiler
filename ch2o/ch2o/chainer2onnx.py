# coding: utf-8

import ast
import gast
import inspect

import numpy as np
from onnx import checker
from onnx import helper
from onnx import TensorProto

import code
import sys
import types

import chainer
import numpy

from ch2o.test_args import dprint
from ch2o.utils import new_tensor, new_sequence, clip_head, ValueReturn, istensor, totensor, Env
from ch2o.links import Link2NodeClass
from ch2o.funcs import Func, Func2NodeClass, Function_Concat, Function_Dummy, castto
from ch2o.builtin_funcs import builtin_functions
from ch2o.value import Value

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


def convert_link(ch, env):
    res = None
    if ch.__class__.__module__[:14] == 'chainer.links.':
        for lk, cl in Link2NodeClass:
            if isinstance(ch, lk):
                res = cl(ch)
                break
        else:
            print('unknown chainer link')
            code.InteractiveConsole({'lk': ch}).interact()
            raise Exception('unknown link', ch)
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
        loenv.vars = args

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


def eval_for(nast, env):
    assert nast.orelse == []
    ite = eval_ast(nast.iter, env)

    if not ite.is_py:
        assert isinstance(nast.target, gast.Name)
        x = nast.target.id

        # 新たなenv を作って、評価中にできた子グラフをもとにする
        localenv = Env(env.module)
        localenv.vars = {}
        localenv.vars.update(env.vars)

        cnt = new_tensor()
        gtx = new_sequence()
        localenv.vars[x] = localenv.calc(
            "OnikuxGenericGetItem",
            inputs=[gtx.name, cnt.name],
        )
        ty = eval_ast(nast.body, localenv)
        assert ty.is_none()

        # 入力側のテンソルを見る。
        in_names = set()
        for no in localenv.nodes:
            in_names = in_names | set(no.input)

        in_names = list(in_names)
        in_closure = {}
        for k, v in env.vars.items():
            if isinstance(v, Value): v = v.value
            if istensor(v) and v.name in in_names:
                in_closure[k] = (v, v)

        compile_retry = False

        # for文の中で値の変更が起こったものは、 in_closure に加える必要がある
        for k in localenv.vars.keys():
            if k not in env.vars.keys():
                # for文の中で新たに変数が定義されることは、とりあえず想定しない。
                # (この場合、forの外に漏れ出していたとしても、Undefined variable にする)
                continue
            if localenv.vars[k] != env.vars[k]:
                if not istensor(env.vars[k]):
                    # この場合、LoopでUpdateしないといけないので
                    # env.vars[k] は tensorでないといけない。
                    # とりあえずコンパイルをやり直しているが、できればやり直さずにしたいですね。
                    env.vars[k] = env.vars[k].to_value_info(env)
                    compile_retry = True

                    """
                    あと、この実装だと、
                    bs = 0
                    cs = 3
                    for s in ilens:
                        cs = s
                        bs = cs
                    のときに、 cs と bs が同じtensorを指していて
                    Loop Operator の出力が同じtensorになってエラーになるので、
                    同じテンソルは出力時に片方dummyにする、などの工夫が必要そう
                    """
                else:
                    in_closure[k] = (env.vars[k], localenv.vars[k])

        if compile_retry:
            return eval_ast(nast, env)

        # ループ内で使われた link パラメータは
        # 1. 外の env にコピーしなければならない
        env.init_tensors.extend(localenv.init_tensors)
        # 2. state としてループ内に持ち込まなければならない
        for init in localenv.init_tensors:
            key = '#' + init.name
            in_closure[key] = (init, init)

        # この時点で
        # in_closure[k] = (a,b) は、
        # 『ループ内で参照されている変数kは
        # ループのbodyの頭にはテンソルaの値であり、
        # ループのbodyの足ではテンソルbの値である』という気持ち

        # print(in_closure)
        # dprint(localenv.nodes)
        # print('ty',ty)

        cond = new_tensor()
        in_closure_values = [(Value(inv).to_value_info(env),
                              Value(outv).to_value_info(env))
                             for inv, outv in in_closure.values()]
        localgraph = helper.make_graph(
            localenv.nodes,
            "Loop_subgraph",
            [cnt, cond, gtx] + [vv[0] for vv in in_closure_values],
            [cond, gtx] + [vv[1] for vv in in_closure_values]
        )

        mtc = env.calc(
            "OnikuxGenericLen",
            inputs=[ite.to_value_info(env).name],
        )

        def dummy():
            return "dummy_" + new_tensor().name

        env.addnode(
            'Loop',
            inputs=[mtc.name, "", ite.to_value_info(env).name] +
            [vv[0].name for vv in in_closure.values()],
            # ループの中でupdateされない変数は出力先はdummyにする
            # updateするならそれでできる新たなテンソルを出力とする
            outputs=[dummy()] + [(dummy() if vv[0] == vv[1] else vv[1].name)
                                 for vv in in_closure_values],
            body=localgraph
        )

        for k, (_, v) in in_closure.items():
            env.vars[k] = v

        return None
    else:
        # とりあえず実際にfor文を回す
        tg = nast.target.id
        env.vars[tg] = None
        for v in ite.value:
            env.vars[tg] = v
            eval_ast(nast.body, env)
            # print('looping',env.vars.keys())

        env.vars.pop(tg)
        return None


def eval_assign(nast, env):
    value = eval_ast(nast.value, env)
    targs = nast.targets
    assert(len(targs) == 1)
    # v,w = 1 も targetsは長さ1のlistになるので len(rargs) != 1 の状況は謎ですね

    # tgとして、下以外に
    # List, ListのIndex, Starred
    # またこれらを再帰的に組み合わせたものが存在しうる

    tg = targs[0]
    if isinstance(tg, gast.Name):
        env.vars[tg.id] = value
    elif isinstance(tg, gast.Tuple):
        assert(isinstance(value.value, tuple))
        value = value.value
        assert(len(tg.elts) == len(value))

        for i, v in enumerate(value):
            env.vars[tg.elts[i].id] = v  # TODO(satos) これこのあと更に再帰的に書く必要あるかも

    elif isinstance(tg, gast.Attribute):
        body = eval_ast(tg.value, env)
        setattr(body.value, tg.attr, value)
    else:
        raise Exception('invalid assing lvalue', targs[0])
    return None


def eval_call(nast, env):
    fn = eval_ast(nast.func, env)
    if not fn.is_py:
        raise TypeError('Expected a callable: %s' % fn.value)
    fn = fn.value

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
    elif isinstance(fn, types.BuiltinFunctionType):
        # BuiltinFunctionType はbuiltinsではなくCで書かれた関数のこととのこと
        fn = builtin_functions[fn.__name__]
    elif fn == range:
        fn = builtin_functions['range']
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
            return res
        elif nast.attr == 'append':
            # TODO(satos) ごまかさない
            assert isinstance(
                nast.value, gast.Name) and nast.value.id in env.vars.keys()
            na = nast.value.id

            # あと、ここのnaがreferenceの場合不正確
            # たとえば
            # x = y
            # x.append(3)
            # のyが更新されないので問題

            def f(args, _, env):
                assert len(args) == 1
                v = args[0].to_tensor(env)
                env.vars[na] = env.calc_seq(
                    'OnikuxSequenceAppend',
                    inputs=[body.to_sequence(env).name, v.name],
                )
                return None

            return Func(f)

        raise Exception('Unimplemented attribute ',
                        nast.attr, ' for tensor')
    return body.get_attribute(nast.attr)


def eval_compare(nast, env):
    le = eval_ast(nast.left, env)
    vs = list(map(lambda x: eval_ast(x, env), nast.comparators))
    # とりあえず定数畳み込みのみにする

    if (istensor(le) or any(map(istensor, vs))):
        raise Exception('unimplemented tensor comparetion')

    res = True
    for op, r in zip(nast.ops, vs):
        if isinstance(op, gast.Eq):
            res = res and (le == r)
        elif isinstance(op, gast.Is):
            res = res and (le is r)
        elif isinstance(op, gast.IsNot):
            res = res and (le is not r)
        elif isinstance(op, gast.Gt):
            res = res and (le > r)
        else:
            raise Exception('unimplemented operator', op)
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
                        body=[tast], orelse=[])

    init = gast.ast_to_gast(ast.parse("v = []")).body[0]
    init.targets[0].id = vn
    tast = [init, tast]

    rv = eval_ast(tast, env)
    assert rv.is_none()
    res = env.vars[vn]
    env.vars.pop(vn)
    return res


def _concat(xs, axis, env):
    assert isinstance(xs, tuple)  # 今のところ tuple 以外は concat できない
    return env.calc(
        "Concat",
        inputs=list(map(lambda x: x.name, xs)),
        axis=axis,
    )


def eval_subscript(nast, env):
    # Subscriptの実装は以下の感じではだめで、
    # コンパイラはシリアライズするだけにして
    # あとはOnikuのほうにお願いすることになりそう

    vs = eval_ast(nast.value, env)
    if isinstance(vs, tuple):
        assert isinstance(nast.slice, gast.Index)
        idx = eval_ast(nast.slice.value, env)
        assert isinstance(idx, int)
        return vs[idx]
    elif isinstance(vs, list):
        raise Exception("unimplemented")

    def unsqueeze(x):
        tx = env.calc(
            'Unsqueeze',
            inputs=[x.name],
            axes=[0]
        )
        return tx

    # sliceがIdxの場合は、Idxのexprにlistが来うる可能性があるのでGatherする
    # Numpyのsliceは闇では...???
    # TODO(satos) Sliceの実装を"ちゃんと"したものにする(現在だと v[0:1,[2,3]] みたいなのは動かない)
    # あと、これだとIndexに副作用が存在する場合にやばい
    if isinstance(nast.slice, gast.Index):
        idx = eval_ast(nast.slice.value, env)
        if istensor(idx):
            res = env.calc(
                'Gather',
                inputs=[vs.name, idx.name],
            )
            return res
        elif isinstance(idx, int):
            # TODO(satos) スライドのためのごまかしのごまかし
            idx = unsqueeze(idx.to_tensor(env))
            res = env.calc(
                'OnikuxGenericGetItem',
                inputs=[vs.name, idx.name],
            )
            return res

    def slice2list(self):
        if isinstance(self, gast.Slice):
            assert self.step is None

            def f(x, v):
                if x is None:
                    return Value(v).to_tensor(env)
                x = eval_ast(x, env)
                if istensor(x):
                    return x
                else:
                    return x.to_tensor(env)
            lower = unsqueeze(f(self.lower, 0))
            # TODO(satos)　その場しのぎっぽいのでどうにかする(けどこれどうにもならないですよね...?)
            upper = unsqueeze(f(self.upper, 2 ** 30))
            squeeze = [False]
        elif isinstance(self, gast.Index):
            idx = eval_ast(self.value, env)
            if isinstance(idx.value, tuple):  # ここにTupleが来うる
                # TODO(satos) もっとうまくやったほうがいいかも
                vs = [gast.Index(gast.NameConstant(value=v)) for v in idx.value]
                lower, upper, squeeze = slice2list(gast.ExtSlice(dims=vs))
            elif not idx.is_py:
                lower = unsqueeze(idx.value)
                ot = totensor(1, env)
                upper = env.calc(
                    "Add",
                    inputs=[idx.to_tensor(env).name, ot.name],
                )
                upper = unsqueeze(upper)
                squeeze = [True]
            else:
                lower = unsqueeze(totensor(idx.value, env))
                upper = unsqueeze(totensor(idx.value+1, env))
                squeeze = [True]
        elif isinstance(self, gast.ExtSlice):
            ds = list(map(slice2list, self.dims))
            lower = _concat(
                tuple(map(lambda x: castto(x[0], TensorProto.INT64, env), ds)), 0, env)
            upper = _concat(
                tuple(map(lambda x: castto(x[1], TensorProto.INT64, env), ds)), 0, env)
            squeeze = sum(map(lambda x: x[2], ds), [])
        else:
            raise Exception(self, " is not Python slice")

        return lower, upper, squeeze
    # print('slice',nast.slice)
    lower, upper, squeeze = slice2list(nast.slice)
    res = new_tensor()
    vs = Value(vs).to_tensor(env)
    lower = Value(lower).to_tensor(env)
    upper = Value(upper).to_tensor(env)
    res = env.calc(
        'DynamicSlice',
        inputs=[vs.name, lower.name, upper.name],
    )

    if any(squeeze):
        res = env.calc(
            'Squeeze',
            inputs=[res.name],
            axes=list(filter(lambda i: squeeze[i], range(len(squeeze))))
        )

    return res


def eval_list(nast, env):
    # Sequenceにしているが、ここはPythonのlistのままにしておきたいとのこと
    # Sequenceにする
    vs = list(map(lambda x: eval_ast(x, env), nast.elts))
    res = env.calc_seq(
        "OnikuxSequenceCreate",
        inputs=[],
    )
    for v in vs:
        v = v.to_tensor(env)
        res = env.calc_seq(
            "OnikuxSequenceAppend",
            inputs=[res.name, v.name],
        )
    return res


def eval_ast(nast, env):
    r = eval_ast_impl(nast, env)
    if (isinstance(r, User_Defined_Function) or
        isinstance(r, User_Defined_Func_In_Link)):
        return r
    return Value(r)


def eval_ast_impl(nast, env):
    if not isinstance(nast, list):
        dprint(gast.dump(nast), env.vars.keys())

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
        b = eval_ast(nast.test, env)
        if b is True:
            return eval_ast(nast.body, env)
        elif b is False:
            return eval_ast(nast.orelse, env)
        else:
            raise Exception('Not constant If is not implemented yet')

    elif isinstance(nast, gast.ListComp):
        return eval_list_comp(nast, env)

    elif isinstance(nast, gast.Subscript):
        return eval_subscript(nast, env)

    elif isinstance(nast, gast.Delete):
        # おのおの単に忘れる
        vs = nast.targets
        for v in vs:
            assert isinstance(v, gast.Name)
            env.vars.pop(v.id)
        return None

    elif isinstance(nast, gast.Name):
        if nast.id in env.vars.keys():
            return env.vars[nast.id]
        elif nast.id in dir(env.module):
            return getattr(env.module, nast.id)
        elif nast.id in dir(builtins):
            return getattr(builtins, nast.id)
        else:
            raise Exception("Undefined name %s" % nast.id)
    elif isinstance(nast, gast.Num):
        return nast.n
    elif isinstance(nast, gast.NameConstant):
        return nast.value
    elif isinstance(nast, gast.Expr):
        return eval_ast(nast.value, env)
    elif isinstance(nast, gast.Str):
        return nast.s
    elif isinstance(nast, gast.Tuple):
        return tuple(map(lambda x: eval_ast(x, env), nast.elts))
    elif isinstance(nast, gast.List):
        return eval_list(nast, env)

    elif isinstance(nast, gast.Return):
        raise ValueReturn(eval_ast(nast.value, env))

    else:
        print('unknown ast')
        code.InteractiveConsole({'nast': nast, 'env': env}).interact()
        raise Exception('unknown ast', nast)

    raise Exception("shouldn't reach here", nast)


def compiler(model):
    # return helper.make_graph([],'dummy',[],[])

    init_id2name(model)
    # code.InteractiveConsole({'mo': model}).interact()
    env = Env(sys.modules[model.__module__])
    molk = User_Defined_Link(model, env)

    input_tensors = []
    for i in range(molk.forward_arglen):  # self 以外
        x = new_tensor()  # ここの次元は不明になる
        input_tensors.append(x)

    v = molk.call(input_tensors, [], env)

    dprint('output_tensors', v)
    if isinstance(v.value, tuple):
        output_tensors = list(v.value)  # ばらしてみる
    else:
        output_tensors = [v]  # とりあえず1tensor

    # print('env.init_tensors ',env.init_tensors)
    input_tensors += env.init_tensors

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
    graph = helper.make_graph(env.nodes,
                              'name_is_unknown_now', input_tensors,
                              outputs_vi,
                              )

    # inputのうち、重みであるものにはinitializerをつける
    # batch_sizeやinput_sizeなどの可変なものはできる限りのそのままで

    dprint(graph)
    # checker.check_graph(graph)
    # oniku独自のノードを使うとcheckできなくなる...
    mo = helper.make_model(graph)

    # print(mo)
    return mo, input_tensors, outputs_vi
