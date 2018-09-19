# coding: utf-8

import ast
import gast
import inspect
import onnx
from onnx import checker
from onnx import helper
from onnx import TensorProto

import code
import sys
import types

import chainer
import numpy

from . test_args import dprint
from . utils import new_tensor, clip_head, ValueReturn, istensor, totensor, Env
from . links import Link2NodeClass
from . funcs import Func, Func2NodeClass, Function_Concat, Function_Dummy, castto
from . builtin_funcs import builtin_functions

import builtins
import six


id2name_list = []


def init_id2name(ch):
    global id2name_list
    id2name_list = []
    for k, v in ch.namedlinks():
        # print('add link',k,v,id(v))
        id2name_list.append((id(v), k.replace('/', '_')))


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
        # code.InteractiveConsole({'v': self.ast.args}).interact()

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

        try:
            eval_ast(self.ast.body, loenv)
            return None
        except ValueReturn as v:
            res = v.value
        return res


class User_Defined_Function(Function_base):
    def __init__(self, func):
        self.func = func
        src = clip_head(inspect.getsource(func))
        dprint(src)
        self.ast = gast.ast_to_gast(ast.parse(src)).body[0]
        assert(isinstance(self.ast, gast.gast.FunctionDef))

    def call(self, args, kwargs, env):
        loenv = env.localenv()
        loenv.links = {}
        loenv.module = sys.modules[self.func.__module__]
        return self.stub_call(args, kwargs, loenv)


class User_Defined_Func_In_Link(Function_base):
    def __init__(self, ch, fn):
        self.ch = ch
        src = clip_head(inspect.getsource(fn))
        dprint(src)
        self.ast = gast.ast_to_gast(ast.parse(src)).body[0]

    def call(self, args, kwargs, env):

        loenv = env.localenv()
        loenv.module = sys.modules[self.ch.__module__]
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
        self.inits = []
        for s, v in ch.namedparams():
            s = s[1:]
            if s.find('/') != -1:
                continue
            t = helper.make_tensor_value_info(
                '_'+s, TensorProto.FLOAT, list(v.shape))
            self.inits.append(t)
            mv = getattr(ch, s)
            setattr(ch, s, t)
            env.restore_funcs.append(lambda: setattr(ch, s, mv))

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


def eval_ast(nast, env):
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
        assert nast.orelse == []
        ite = eval_ast(nast.iter, env)

        if istensor(ite):
            assert isinstance(nast.target, gast.Name)
            x = nast.target.id

            # 新たなenv を作って、評価中にできた子グラフをもとにする
            localenv = Env()
            localenv.vars = {}
            for k, v in env.vars.items():
                localenv.vars[k] = v
            localenv.module = env.module
            tx = new_tensor()
            localenv.vars[x] = tx
            ty = eval_ast(nast.body,  localenv)
            assert ty is None

            cnt = new_tensor()
            gtx = new_tensor()
            localenv.addnode(
                "OnikuxGenericGetItem",
                inputs=[gtx.name, cnt.name], outputs=[tx.name],
            )

            # 入力側のテンソルを見る。
            closure = set()
            for no in localenv.nodes:
                closure = closure | set(no.input)

            cnames = list(closure)
            in_closure = {}
            for k, v in env.vars.items():
                if istensor(v) and v.name in cnames:
                    in_closure[k] = (v, v)

            # graph内で出力されるテンソルは環境を上書きしないといけない
            closure = set()
            for no in localenv.nodes:
                closure = closure | set(no.output)

            cnames = list(closure)
            # 名前しか得られないのでテンソルを得る
            # 生きているのはvarsで参照できるやつだけ...だと思う
            for k, v in localenv.vars.items():
                if istensor(v) and v.name in cnames:
                    if not (k in in_closure.keys()):
                        """
                        if k in env.vars.keys():
                            # 実はテンソルになる必要があったやつなので、再登録する
                            fv = env.vars[k]
                            fv = totensor(fv,env)
                            in_closure[k] = (fv,fv)  
                        else:
                            # for文の中で新たに変数が定義されることは、とりあえず想定しない。
                            # (この場合、Undefined variable にする)
                            continue
                        """
                        # これだとgraphを再評価する必要があるのでだめ
                        # TODO(satos) どうにかする
                        continue

                    fv, _ = in_closure[k]
                    in_closure[k] = (fv, v)

            # print(in_closure)
            # dprint(localenv.nodes)
            # print('ty',ty)
            cond = new_tensor()
            localgraph = helper.make_graph(
                localenv.nodes,
                "Loop_subgraph",
                [cnt, cond, gtx] + [vv[0] for vv in in_closure.values()],
                [cond, gtx] + [vv[1] for vv in in_closure.values()]
            )

            mtc = new_tensor()
            env.addnode(
                "OnikuxGenericLen",
                inputs=[ite.name], outputs=[mtc.name]
            )

            def dummy():
                return "dummy_" + new_tensor().name

            env.addnode(
                'Loop',
                inputs=[mtc.name, "", ite.name] +
                [vv[0].name for vv in in_closure.values()],
                # ほかのと名前が衝突しないようにする
                outputs=[dummy()] + [(dummy() if vv[0] == vv[1] else vv[1].name)
                                     for vv in in_closure.values()],
                body=localgraph
            )

            for k, (_, v) in in_closure.items():
                env.vars[k] = v

            return None
        else:
            # とりあえず実際にfor文を回す
            tg = nast.target.id
            env.vars[tg] = None
            for v in ite:
                env.vars[tg] = v
                eval_ast(nast.body, env)
                # print('looping',env.vars.keys())

            env.vars.pop(tg)
            return None
    elif isinstance(nast, gast.Assign):
        value = eval_ast(nast.value, env)
        targs = nast.targets
        assert(len(targs) == 1)

        tg = targs[0]
        if isinstance(tg, gast.Name):
            env.vars[tg.id] = value
        elif isinstance(tg, gast.Tuple):
            assert(isinstance(value, tuple))
            assert(len(tg.elts) == len(value))

            for i, v in enumerate(value):
                env.vars[tg.elts[i].id] = v  # TODO(satos) これこのあと更に再帰的に書く必要あるかも

        elif isinstance(tg, gast.Attribute):
            body = eval_ast(tg.value, env)
            setattr(body, tg.attr, value)
        else:
            raise Exception('invalid assing lvalue', targs[0])
        return None

    elif isinstance(nast, gast.AugAssign):
        ca = gast.Assign(targets=[nast.target], value=gast.BinOp(
            left=nast.target, op=nast.op, right=nast.value))
        return eval_ast(ca, env)

    elif isinstance(nast, gast.Call):
        fn = eval_ast(nast.func, env)
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
        for fr, to in Func2NodeClass:
            if fr == fn:
                return to.call(args, keywords, env)

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

    elif isinstance(nast, gast.UnaryOp):
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
            return opfun(v)
        else:
            raise Exception("Unimplemented yet")

    elif isinstance(nast, gast.BinOp):
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

        if not istensor(lv) and not istensor(rv):
            return opfun(lv, rv)

        lv = totensor(lv, env)
        rv = totensor(rv, env)

        env.addnode(
            optype,
            inputs=[lv.name, rv.name], outputs=[res.name],
        )

        if isfloor:
            r = new_tensor()
            env.addnode(
                "Floor",
                inputs=[res.name], outputs=[r.name],
            )
            res = r

        return res

    elif isinstance(nast, gast.BoolOp):
        vs = list(map(lambda x: eval_ast(x, env), nast.values))
        res = new_tensor(['TODO'])
        if isinstance(nast.op, gast.And):
            def opfun(v): return all(v)
        else:
            raise Exception('unknown operator', nast.op)

        if not any(map(istensor, vs)):
            return opfun(vs)

        raise Exception('Unimplemented BoolOp for tensor', nast)

    elif isinstance(nast, gast.Attribute):
        body = eval_ast(nast.value, env)

        if istensor(body):
            if nast.attr == 'shape':
                res = new_tensor()
                env.addnode(
                    'Shape',
                    inputs=[body.name], outputs=[res.name],
                )
                return res
            elif nast.attr == 'append':
                # TODO(satos) ごまかさない()
                assert isinstance(
                    nast.value, gast.Name) and nast.value.id in env.vars.keys()
                na = nast.value.id

                def f(args, _, env):
                    assert len(args) == 1
                    v = args[0]
                    res = new_tensor()
                    env.addnode(
                        'OnikuxSequenceAppend',
                        inputs=[body.name, v.name], outputs=[res.name]
                    )
                    env.vars[na] = res
                return Func(f)

            raise Exception('Unimplemented attribute ',
                            nast.attr, ' for tensor')
        return getattr(body, nast.attr)

    elif isinstance(nast, gast.Compare):
        le = eval_ast(nast.left, env)
        vs = list(map(lambda x: eval_ast(x, env), nast.comparators))
        # とりあえず定数畳み込みのみにする

        if (istensor(le) or any(map(istensor, vs))):
            # TODO(satos) めちゃ緊急回避
            if nast.left.id == 'dec_z':
                return False

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

    elif isinstance(nast, gast.If):
        b = eval_ast(nast.test, env)
        if b is True:
            return eval_ast(nast.body, env)
        elif b is False:
            return eval_ast(nast.orelse, env)
        else:
            raise Exception('Not constant If is not implemented yet')

    elif isinstance(nast, gast.ListComp):
        # [ なんやかや for x in xs] 形式のものを Loopで対応する
        assert len(nast.generators) == 1
        gen = nast.generators[0]
        assert len(gen.ifs) == 0
        assert gen.is_async == 0

        xs = eval_ast(gen.iter, env)
        assert isinstance(gen.target, gast.Name)
        x = gen.target.id

        # 新たなenv を作って、評価中にできた子グラフをもとにする
        localenv = Env()
        localenv.vars = {}
        for k, v in env.vars.items():
            localenv.vars[k] = v
        localenv.module = env.module
        tx = new_tensor()
        localenv.vars[x] = tx
        ty = eval_ast(nast.elt,  localenv)

        cnt = new_tensor()
        gtx = new_tensor()
        localenv.addnode(
            "OnikuxGenericGetItem",
            inputs=[gtx.name, cnt.name], outputs=[tx.name],
        )

        ty_init = new_tensor()
        tty = new_tensor()
        localenv.addnode(
            "OnikuxSequenceAppend",
            inputs=[ty_init.name, ty.name], outputs=[tty.name],
        )
        ty = tty

        # graph内で参照されるテンソルは入力として与えないといけない。
        closure = set()
        removes = set()
        for no in localenv.nodes:
            closure = closure | set(no.input)
            removes = removes | set(no.output)

        closure = closure - removes
        cnames = list(closure)
        closure = []
        # 名前しか得られないのでテンソルを得る
        # 生きているのはvarsで参照できるやつだけ...だと思う
        for _, v in env.vars.items():
            if istensor(v) and v.name in cnames:
                closure.append(v)

        cnames = [x.name for x in closure]

        # dprint(localenv.nodes)
        # print('ty',ty)
        cond = new_tensor()
        localgraph = helper.make_graph(
            localenv.nodes,
            "Loop_subgraph",
            [cnt, cond, gtx] + closure + [ty_init],
            [cond, gtx] + closure + [ty]
        )

        mtc = new_tensor()
        """
        v = new_tensor()
        env.addnode(
            "Shape",
            inputs=[xs.name], outputs=[v.name]
        )

        env.addnode(
            "Gather",
            inputs=[v.name, totensor(0, env).name], outputs=[mtc.name],
            axis=0
        )
        """

        nullseq = new_tensor()
        env.addnode(
            "OnikuxSequenceCreate",
            inputs=[], outputs=[nullseq.name]
        )

        env.addnode(
            "OnikuxGenericLen",
            inputs=[xs.name], outputs=[mtc.name]
        )

        dummy_name = ["dummy_" +
                      new_tensor().name for _ in range(len(cnames)+1)]
        res = new_tensor()
        env.addnode(
            'Loop',
            inputs=[mtc.name, "", xs.name] + cnames + [nullseq.name],
            # ほかのと名前が衝突しないようにする
            outputs=dummy_name+[res.name],
            body=localgraph
        )

        return res

    elif isinstance(nast, gast.Subscript):
        vs = eval_ast(nast.value, env)
        if isinstance(vs, tuple):
            assert isinstance(nast.slice, gast.Index)
            idx = eval_ast(nast.slice.value, env)
            assert isinstance(idx, int)
            return vs[idx]
        elif isinstance(vs, list):
            raise Exception("unimplemented")

        def unsqueeze(x):
            tx = new_tensor()
            env.addnode(
                'Unsqueeze',
                inputs=[x.name], outputs=[tx.name],
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
                res = new_tensor()
                env.addnode(
                    'Gather',
                    inputs=[vs.name, idx.name],
                    outputs=[res.name]
                )
                return res
            elif isinstance(idx, int):
                # TODO(satos) スライドのためのごまかしのごまかし
                res = new_tensor()
                idx = unsqueeze(totensor(idx, env))
                env.addnode(
                    'OnikuxGenericGetItem',
                    inputs=[vs.name, idx.name],
                    outputs=[res.name]
                )
                return res

        def slice2list(self):
            if isinstance(self, gast.Slice):
                assert self.step is None

                def f(x, v):
                    if x is None:
                        return totensor(v, env)
                    x = eval_ast(x, env)
                    if istensor(x):
                        return x
                    else:
                        return totensor(x, env)
                lower = unsqueeze(f(self.lower, 0))
                # TODO(satos)　その場しのぎっぽいのでどうにかする(けどこれどうにもならないですよね...?)
                upper = unsqueeze(f(self.upper, 2 ** 30))
                squeeze = [False]
            elif isinstance(self, gast.Index):
                idx = eval_ast(self.value, env)
                if isinstance(idx, tuple):  # ここにTupleが来うる
                    # TODO(satos) もっとうまくやったほうがいいかも
                    vs = [gast.Index(gast.NameConstant(value=v)) for v in idx]
                    lower, upper, squeeze = slice2list(gast.ExtSlice(dims=vs))
                elif istensor(idx):
                    lower = unsqueeze(idx)
                    ot = totensor(1, env)
                    upper = new_tensor()
                    env.addnode(
                        "Add",
                        inputs=[idx.name, ot.name], outputs=[upper.name],
                    )
                    upper = unsqueeze(upper)
                    squeeze = [True]
                else:
                    lower = unsqueeze(totensor(idx, env))
                    upper = unsqueeze(totensor(idx+1, env))
                    squeeze = [True]
            elif isinstance(self, gast.ExtSlice):
                ds = list(map(slice2list, self.dims))
                lower = Function_Concat().call(
                    [list(map(lambda x: castto(x[0], TensorProto.INT32, env), ds))], {'axis': 0}, env)
                upper = Function_Concat().call(
                    [list(map(lambda x: castto(x[1], TensorProto.INT32, env), ds))], {'axis': 0}, env)
                squeeze = sum(map(lambda x: x[2], ds), [])
            else:
                raise Exception(self, " is not Python slice")

            return lower, upper, squeeze
        # print('slice',nast.slice)
        lower, upper, squeeze = slice2list(nast.slice)
        res = new_tensor()
        env.addnode(
            'DynamicSlice',
            inputs=[vs.name, lower.name, upper.name],
            outputs=[res.name]
        )

        if any(squeeze):
            r = new_tensor()
            env.addnode(
                'Squeeze',
                inputs=[res.name], outputs=[r.name],
                axes=list(filter(lambda i: squeeze[i], range(len(squeeze))))
            )
            res = r

        return res

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
        # Sequenceにする
        vs = list(map(lambda x: eval_ast(x, env), nast.elts))
        res = new_tensor()
        env.addnode(
            "OnikuxSequenceCreate",
            inputs=[], outputs=[res.name]
        )
        for v in vs:
            v = totensor(v, env)
            tr = new_tensor()
            env.addnode(
                "OnikuxSequenceAppend",
                inputs=[res.name, v.name], outputs=[tr.name]
            )
            res = tr
        return res
    elif isinstance(nast, gast.Return):
        raise ValueReturn(eval_ast(nast.value, env))
    else:
        print('unknown ast')
        code.InteractiveConsole({'nast': nast, 'env': env}).interact()
        raise Exception('unknown ast', nast)

    raise Exception("shouldn't reach here", nast)


def chainer2onnx(model, forward):
    # return helper.make_graph([],'dummy',[],[])

    init_id2name(model)
    # code.InteractiveConsole({'mo': model}).interact()
    env = Env()
    molk = User_Defined_Link(model, env)

    input_tensors = []
    for i in range(molk.forward_arglen):  # self 以外
        x = new_tensor()  # ここの次元は不明になる
        input_tensors.append(x)

    # forward = molk.forward
    env.module = sys.modules[model.__module__]
    # for k,v in zip(map(lambda x: x.id,forward.args.args),[model] + input_tensors):
    #    env.vars[k] = v

    v = molk.call(input_tensors, [], env)

    dprint('output_tensors', v)
    if isinstance(v, tuple):
        output_tensors = list(v)  # ばらしてみる
    else:
        output_tensors = [v]  # とりあえず1tensor

    #print('env.init_tensors ',env.init_tensors)
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

    graph = helper.make_graph(env.nodes,
                              'name_is_unknown_now', input_tensors,
                              output_tensors
                              )

    # inputのうち、重みであるものにはinitializerをつける
    # batch_sizeやinput_sizeなどの可変なものはできる限りのそのままで

    dprint(graph)
    # exit(0)
    # checker.check_graph(graph)
    # oniku独自のノードを使うとcheckできなくなる...
    # Scanがあると、 No Schema registered for Scan with domain_version of 4 でだめぽい
    mo = helper.make_model(graph)

    # print(mo)
    return mo, input_tensors, output_tensors
