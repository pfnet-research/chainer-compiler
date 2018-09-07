# coding: utf-8

import ast
import gast
import inspect
import onnx
from onnx import checker
from onnx import helper

import code
import sys
import types

import chainer
import numpy

from . test_args import dprint
from . utils import new_tensor, clip_head, ValueReturn, istensor
from . links import Link2NodeClass
from . funcs import Func, Func2NodeClass
from . xp_numpy import xp_attrs, np_attrs
from . builtin_funcs import builtin_functions

import builtins
import six


def initLinks(model, parentname):
    links = {}

    for na, ch in model.__dict__.items():
        # print(na,ch,issubclass(ch.__class__,chainer.Chain),ch.__class__.__module__)
        # code.InteractiveConsole({'na': na,'ch': ch}).interact()
        if issubclass(ch.__class__, chainer.link.Link):
            if ch.__class__.__module__[:14] == 'chainer.links.':
                for lk, cl in Link2NodeClass:
                    if isinstance(ch, lk):
                        links[na] = cl(ch, parentname)
                        break
                else:
                    print('unknown chainer link')
                    code.InteractiveConsole({'lk': ch}).interact()
                    raise Exception('unknown link', ch)
            else:
                # User Defined link
                links[na] = User_Defined_Link(ch, parentname)
    return links


class Attr(object):
    def __init__(self, d, ch=None):
        self.dic = d
        self.ch = ch

    def update(self, d):
        self.dic.update(d)

    def get_attr(self, k):
        if k in self.dic.keys():
            return self.dic[k]
        elif (self.ch is not None) and (k in dir(self.ch)) and (callable(getattr(self.ch, k))):
            return Func(getattr(self.ch, k))
        else:
            raise Exception(self.ch, 'has no attr ', k)

    def set_attr(self, k, v):
        self.dic[k] = v


class Function_base(object):
    def stub_call(self, args, kwargs, loenv):

        # code.InteractiveConsole({'v': self.ast.args}).interact()

        astargs = list(map(lambda x: x.id, self.ast.args.args))
        args = dict(zip(astargs, args))

        defs = self.ast.args.defaults
        d = len(astargs) - len(args.keys())
        if d > 0:
            for i, v in enumerate(defs[::-1][:d]):
                args.update({astargs[-i-1]: v})

        args.update(kwargs)
        # args.update()

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

        src = clip_head(inspect.getsource(func))
        dprint(src)
        self.ast = gast.ast_to_gast(ast.parse(src)).body[0]
        assert(isinstance(self.ast, gast.gast.FunctionDef))

    def call(self, args, kwargs, env):
        # code.InteractiveConsole({'nast':self.ast}).interact()

        loenv = env.localenv()
        loenv.links = {}
        loenv.module = env.module

        return self.stub_call(args, kwargs, loenv)


class User_Defined_Func_In_Link(Function_base):
    def __init__(self, attrs, links, module, ast, funname):
        self.attrs = attrs
        self.links = links
        self.module = module
        self.ast = ast

        # for debuging
        self.funname = funname

    def call(self, args, kwargs, env):
        print('calling', self.funname)
        # code.InteractiveConsole({'nast':self.ast}).interact()

        loenv = env.localenv()
        loenv.links = self.links
        loenv.module = self.module
        args = [self.attrs] + args

        return self.stub_call(args, kwargs, loenv)


class User_Defined_Link(object):
    def __init__(self, ch, parentname):
        self.name = parentname + ('' if ch.name is None else '_' + ch.name)
        # print('UserDefined',ch)
        # code.InteractiveConsole({'ch': ch}).interact()

        self.attrs = Attr({}, ch)
        self.links = initLinks(ch, self.name)
        self.module = sys.modules[ch.__module__]

        # print(gast.ast_to_gast(ast.parse(clip_head(inspect.getsource(ch.__class__)))).body[0].body)

        src = clip_head(inspect.getsource(ch.__class__))
        dprint(src)

        ast_list = gast.ast_to_gast(ast.parse(src)).body[0].body
        # code.InteractiveConsole({'v': ast_list}).interact()
        funcs = {}
        for func in ast_list:
            if not isinstance(func, gast.gast.FunctionDef):
                continue

            if func.name == 'forward':
                # このへんもkwargsの扱いとかどうにかしないと
                self.forward_arglen = len(func.args.args)-1

            funcs[func.name] = User_Defined_Func_In_Link(
                self.attrs, self.links, self.module, func, self.name + "#" + ch.__class__.__name__ + "@" + func.name)

        self.attrs.update(vars(ch))
        self.attrs.update(self.links)
        self.attrs.update({
            'xp': Attr(xp_attrs),
        })
        self.attrs.update(funcs)

        # print(ch.name,self.attrs.dic.keys())
        self.get_attr = self.attrs.get_attr
        # print(self.attrs.dic)

        self.call = self.get_attr('forward').call

        # print(self.attrs.dic)
        # exit(0)

    def init_tensors(self):
        res = []
        for l in self.links.values():
            res += l.init_tensors()
        return res


class User_Defined_Class(object):
    def __init__(self, classtype):
        self.attrs = Attr({})

        src = clip_head(inspect.getsource(classtype))
        dprint(src)

        ast_list = gast.ast_to_gast(ast.parse(src)).body[0].body
        # code.InteractiveConsole({'v': classtype}).interact()
        self.module = sys.modules[classtype.__module__]

        funcs = {}
        for func in ast_list:
            if not isinstance(func, gast.gast.FunctionDef):
                continue

            funcs[func.name] = User_Defined_Func_In_Link(
                self.attrs, {}, self.module, func, classtype.__name__ + "@" + func.name)

        self.attrs.update(funcs)

        if issubclass(classtype, chainer.FunctionNode):
            def applyfunc(args, kwargs, env):
                # print('apply arg',args)
                return self.get_attr('forward').call(args, kwargs, env)

            self.attrs.update({
                'apply': Func(applyfunc),
                # TODO(satos) これbackward側に何か伝える必要がありそう
                'retain_inputs': Func(lambda _, __, ___: None),
            })

        # print(ch.name,self.attrs.dic.keys())
        self.get_attr = self.attrs.get_attr
        # print(self.attrs.dic)

        self.call = self.get_attr('forward').call

        def f(args, kwargs, env):
            self.get_attr('__init__').call(args, kwargs, env)
            return self

        self.init_wrapper = Func(f)


class Env(object):
    def __init__(self):
        self.vars = {}
        self.nodes = []

    def localenv(self):
        res = Env()
        res.nodes = self.nodes  # こっちはglobalに共通でないといけない
        return res

    def addnode(self, *args, **kwargs):
        self.nodes.append(
            helper.make_node(*args, **kwargs)
        )


import logging


def islogging(s, env):
    # dprint('check logging',gast.dump(s), env.vars.keys())
    return (
        isinstance(s, gast.Expr) and
        isinstance(s.value, gast.Call) and
        isinstance(s.value.func, gast.Attribute) and
        isinstance(eval_ast(s.value.func.value, env), logging.__class__)
    )


def eval_ast(nast, env):
    if not isinstance(nast, list):
        dprint(gast.dump(nast), env.vars.keys())

    if isinstance(nast, list):
        # 逐次実行
        for s in nast:
            if islogging(s, env):
                continue
            eval_ast(s, env)
    elif isinstance(nast, gast.For):
        # とりあえず実際にfor文を回す
        tg = nast.target.id
        env.vars[tg] = None
        ite = eval_ast(nast.iter, env)

        if istensor(ite):

            tx = new_tensor()
            env.vars[tg] = tx
            ty = eval_ast(nast.body, env)

            # TODO(satos) あとまわします
            return new_tensor()
        else:
            for v in ite:
                env.vars[tg] = v
                eval_ast(nast.body, env)
                # print('looping',env.vars.keys())

            # print('finish loop',env.vars.keys())
            env.vars.pop(tg)

    elif isinstance(nast, gast.Assign):
        value = eval_ast(nast.value, env)
        targs = nast.targets
        assert(len(targs) == 1)

        tg = targs[0]
        if isinstance(tg, gast.Name):
            env.vars[tg.id] = value
        elif isinstance(tg, gast.Tuple):
            # code.InteractiveConsole({'tg': tg,'v': value}).interact()
            assert(isinstance(value, tuple))
            assert(len(tg.elts) == len(value))

            for i, v in enumerate(value):
                env.vars[tg.elts[i].id] = v  # これこのあと更に再帰的に書く必要あるかも

        elif isinstance(tg, gast.Attribute):
            body = eval_ast(tg.value, env)
            body.set_attr(tg.attr, value)
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

        if isinstance(fn, types.FunctionType):
            fn = User_Defined_Function(fn)
        elif isinstance(fn, types.BuiltinFunctionType):
            fn = builtin_functions[fn.__name__]
        elif fn == range:
            fn = builtin_functions['range']
        elif isinstance(fn, type):
            # なにがしかのinstance 作成
            assert fn.__module__ != 'builtins'

            fn = User_Defined_Class(fn).init_wrapper

        # print('funccall',fn,args,keywords,env)
        return fn.call(args, keywords, env)

    elif isinstance(nast, gast.UnaryOp):
        v = eval_ast(nast.operand, env)
        res = new_tensor()
        if isinstance(nast.op, gast.USub):
            # optype = "Sub"
            def opfun(x): return -x
        elif isinstance(nast.op, gast.Not):
            # optype = "Sub"
            def opfun(x): return not x
        else:
            raise Exception('unknown operator', nast.op)

        if not istensor(v):
            return opfun(v)
        else:
            raise Exception("Unimplemented")

    elif isinstance(nast, gast.BinOp):
        lv = eval_ast(nast.left, env)
        rv = eval_ast(nast.right, env)
        res = new_tensor(['TODO'])
        isfloor = False
        if isinstance(nast.op, gast.Add):
            optype = "Add"

            def opfun(a, b): return a + b
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

        def totensor(x):
            if istensor(x):
                return x
            res = new_tensor()

            env.addnode(
                'Constant',
                inputs=[], outputs=[res.name],
                value=onnx.helper.make_tensor(
                    name="hoge",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[],
                    vals=[x],
                )
            )
            return res

        lv = totensor(lv)
        rv = totensor(rv)

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

        # code.InteractiveConsole({'lv': lv, 'rv': rv}).interact()
        if not any(map(istensor, vs)):
            return opfun(vs)

        raise Exception('Unimplemented BoolOp for tensor', nast)

    elif isinstance(nast, gast.Attribute):
        body = eval_ast(nast.value, env)
        # code.InteractiveConsole({'nast':nast,'env': env, 'body':body}).interact()

        if isinstance(body, chainer.variable.Parameter):
            if nast.attr == 'shape':
                return body.shape
        elif body == chainer.functions:
            v = getattr(body, nast.attr)
            for f, c in Func2NodeClass:
                if v == f:
                    return c()
            else:
                raise Exception('unknown function', nast.attr)
        elif body == numpy:
            # raise Exception('unknown function',body, nast.attr)
            return np_attrs[nast.attr]
        elif istensor(body):
            if nast.attr == 'shape':
                res = new_tensor()
                env.addnode(
                    'Shape',
                    inputs=[body.name], outputs=[res.name],
                )
                return res
            elif nast.attr == '__add__':
                # TODO(satos) 応急処置なのであとで消す
                return Func(lambda _,__,___: new_tensor())
        elif body == chainer.backends.cuda:
            if nast.attr == 'to_cpu':
                # TODO(satos) テンソルの位置についてCPUを通ったかどうかを残す
                return Func(lambda x, _, __: x[0])
            elif nast.attr == 'get_array_module':
                # とりあえずnumpyを返していいんでは
                return Func(lambda ___, _, __: numpy)
        elif body == six:
            if nast.attr == 'moves':
                return six.moves
        elif body == six.moves:
            if nast.attr == 'range':
                return builtin_functions['range']
        # TODO(satos) どうすんのこれ(とりあえずhttps://github.com/espnet/espnet/blob/master/src/nets/deterministic_embed_id.py#L43) のif文を通んないようにする
        elif body == chainer.utils.type_check:
            if nast.attr == 'same_types':
                return Func(lambda _, __, ___: True)

        elif body == chainer:
            if nast.attr == 'is_debug':
                # とりあえずfalseでは
                return Func(lambda ___, _, __: False)

        else:
            dprint('getattr', body, nast.attr)
            return body.get_attr(nast.attr)

        raise Exception('value', body, 'attribute',
                        nast.attr, 'is not imlemented yet')

    elif isinstance(nast, gast.Compare):
        le = eval_ast(nast.left, env)
        vs = list(map(lambda x: eval_ast(x, env), nast.comparators))
        # とりあえず定数畳み込みのみにする
        assert not (istensor(le) or any(map(istensor, vs)))

        res = True
        for op, r in zip(nast.ops, vs):
            # code.InteractiveConsole({'op': op}).interact()
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
        # [ なんやかや for x in xs] 形式のものを Scanで対応する
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

        # Scan は map の map なので、一旦[x]で包んでかけてしまう

        print(localenv.nodes)
        localgraph = helper.make_graph(
            localenv.nodes,
            "Scan_subgraph", [tx], [ty]
        )

        txs = new_tensor()
        env.addnode(
            'Unsqueeze',
            inputs=[xs.name], outputs=[txs.name],
            axes=[0]
        )

        bres = new_tensor()
        env.addnode(
            'Scan',
            inputs=[txs.name], outputs=[bres.name],
            body=localgraph,
            num_scan_inputs=1
        )

        res = new_tensor()
        env.addnode(
            'Squeeze',
            inputs=[bres.name], outputs=[res.name],
            axes=[0]
        )
        return res

    elif isinstance(nast, gast.Subscript):
        vs = eval_ast(nast.value, env)

        def slice2list(self):
            if isinstance(self, gast.Slice):
                assert self.step is None
                lower = eval_ast(self.lower, env)
                upper = eval_ast(self.upper, env)
                lower = [0] if lower is None else [lower]
                upper = [-1] if upper is None else [upper]
                squeeze = [False]
            elif isinstance(self, gast.Index):
                idx = eval_ast(self.value, env)
                lower = [idx]
                upper = [idx+1]
                squeeze = [True]
            elif isinstance(self, gast.ExtSlice):
                ds = map(slice2list, self.dims)
                lower = list(map(lambda x, _, __: x, ds))
                upper = list(map(lambda _, x, __: x, ds))
                squeeze = list(map(lambda _, __, x: x, ds))
            else:
                raise Exception(self, " is not Python slice")

            return lower, upper, squeeze

        """
        print('vs',vs)
        if not istensor(vs):
           sls = slice2list(nast.slice)
           print(sls)
           raise Exception("Unimplemented")
        """

        vs = new_tensor()

        # TODO(satos) このままだといろいろまずいきがする(合わせるとか以前に、indexが可変でない)
        # のでどうにかしたい

        res = new_tensor()
        env.addnode(
            'Donika',
            inputs=[vs.name], outputs=[res.name],
        )
        return res

        lower, upper, squeeze = slice2list(nast.slice)
        res = new_tensor()
        env.addnode(
            'Slice',
            inputs=[vs.name], outputs=[res.name],
            starts=lower,
            ends=upper
        )

        if any(squeeze):
            r = new_tensor()
            env.addnode(
                'SqueezeTekina',
                inputs=[r.name], outputs=[res.name],
                # TODO(satos) このままだといろいろまずいきがする(合わせるとか以前に、indexが可変でない)
                axes=[0]
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
        if nast.id == 'print':  # とりあえずの実装なのであとでもっとうまくやる
            return Func(lambda _, __, ___: None)
        elif nast.id in env.vars.keys():
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
        return list(map(lambda x: eval_ast(x, env), nast.elts))
    elif isinstance(nast, gast.Return):
        raise ValueReturn(eval_ast(nast.value, env))
    else:
        print('unknown ast')
        code.InteractiveConsole({'nast': nast, 'env': env}).interact()
        raise Exception('unknown ast', nast)


def chainer2onnx(model, forward):
    # return helper.make_graph([],'dummy',[],[])

    # code.InteractiveConsole({'mo': model}).interact()
    molk = User_Defined_Link(model, '')

    input_tensors = []
    for i in range(molk.forward_arglen):  # self 以外
        x = new_tensor(['batch_size%d' % i, 'input_size%d' % i])
        input_tensors.append(x)

    env = Env()
    v = molk.call(input_tensors, [], env)  # keywordsはとりあえず空

    dprint('output_tensors', v)
    if isinstance(v, tuple):
        output_tensors = list(v)  # ばらしてみる
    else:
        output_tensors = [v]  # とりあえず1tensor

    input_tensors += molk.init_tensors()

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
