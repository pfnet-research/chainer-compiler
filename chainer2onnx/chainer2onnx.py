# coding: utf-8

import ast
import gast
import inspect
import onnx
from onnx import checker
from onnx import helper

import code
import copy
import sys

import chainer

from . test_args import get_test_args
from . utils import new_tensor, clip_head, ValueReturn
from . links_funcs import Func, Func2NodeClass, Link2NodeClass


class User_Defined_Link(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + ('' if ch.name is None else '_' + ch.name)
        # code.InteractiveConsole({'ch': ch}).interact()

        src = clip_head(inspect.getsource(ch.forward))
        # if not get_test_args().quiet:
        #    print(src)
        sl.ast = gast.ast_to_gast(ast.parse(src)).body[0]
        assert(isinstance(sl.ast, gast.gast.FunctionDef))

        sl.links = initLinks(ch, sl.name)
        sl.module = sys.modules[ch.__module__]
        sl.forward_args = list(map(lambda x: x.id, sl.ast.args.args))

        sl.attrs = {'children': Func(lambda _, __, ___: sl.links.values())}
        sl.attrs.update(sl.links)

    def call(sl, args, _, env):
        # 自身のforwardを呼ぶ
        # print('calling',sl.name,'with',args)
        # code.InteractiveConsole({'nast':sl.ast}).interact()

        loenv = env.localenv()

        args = [sl.attrs] + args
        assert(len(sl.forward_args) == len(args))
        loenv.vars = dict(zip(sl.forward_args, args))

        loenv.links = sl.links
        loenv.module = sl.module
        # print('name', sl.name, 'modules', sl.module)

        try:
            eval_ast(sl.ast.body, loenv)
            raise Exception('return not found')
        except ValueReturn as v:
            res = v.value
        return res

    def init_tensors(sl):
        res = []
        for l in sl.links.values():
            res += l.init_tensors()
        return res


def initLinks(model, parentname):
    links = {}
    for ch in model.children():
        if ch.__class__.__module__[:13] == 'chainer.links':
            for lk, cl in Link2NodeClass:
                if isinstance(ch, lk):
                    links[ch.name] = cl(ch, parentname)
                    break
            else:
                print('unknown chainer link')
                code.InteractiveConsole({'lk': ch}).interact()
                raise Exception('unknown link', ch)
        else:
            # User Defined link
            links[ch.name] = User_Defined_Link(ch, parentname)

    return links


class Env(object):
    def __init__(sl):
        sl.vars = {}
        sl.nodes = []

    def localenv(sl):
        res = Env()
        res.nodes = sl.nodes  # こっちはglobalに共通でないといけない
        return res

    def addnode(sl, *args, **kwargs):
        sl.nodes.append(
            helper.make_node(*args, **kwargs)
        )


def eval_ast(nast, env):
    print(nast, env.vars.keys())
    if isinstance(nast, list):
        # 逐次実行
        for s in nast:
            eval_ast(s, env)
    elif isinstance(nast, gast.For):
        # とりあえず実際にfor文を回す
        tg = nast.target.id
        for v in eval_ast(nast.iter, env):
            env.vars[tg] = v
            eval_ast(nast.body, env)
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
        else:
            raise Exception('invalid assing lvalue', targs[0])

    elif isinstance(nast, gast.Call):
        fn = eval_ast(nast.func, env)
        args = list(map(lambda x: eval_ast(x, env), nast.args))
        keywords = dict(
            map(lambda x: (x.arg, eval_ast(x.value, env)), nast.keywords))
        return fn.call(args, keywords, env)
    elif isinstance(nast, gast.UnaryOp):
        v = eval_ast(nast.operand, env)
        res = new_tensor()
        if isinstance(nast.op, gast.USub):
            # optype = "Sub"
            def f(x): return -x
        else:
            raise Exception('unknown operator', nast.op)

        def istensor(x):
            return isinstance(x, onnx.onnx_ONNX_NAMESPACE_ml_pb2.ValueInfoProto)

        if not istensor(v):
            return f(v)
        else:
            raise Exception("Unimplemented")

    elif isinstance(nast, gast.BinOp):
        lv = eval_ast(nast.left, env)
        rv = eval_ast(nast.right, env)
        res = new_tensor(['TODO'])
        if isinstance(nast.op, gast.Add):
            optype = "Add"
        elif isinstance(nast.op, gast.Mult):
            optype = "Mul"
        else:
            raise Exception('unknown operator', nast.op)

        # code.InteractiveConsole({'lv': lv, 'rv': rv}).interact()
        def istensor(x):
            return isinstance(x, onnx.onnx_ONNX_NAMESPACE_ml_pb2.ValueInfoProto)

        if not istensor(lv) and not istensor(rv):
            if optype == "Add":
                return lv + rv
            elif optype == "Mul":
                return lv * rv
            else:
                raise Exception('unknown operator', nast.op)

        def totensor(x):
            if istensor(x):
                return x
            res = new_tensor(['TODO'])

            env.nodes.append(
                helper.make_node(
                    'Constant',
                    inputs=[], outputs=[res.name],
                    value=onnx.helper.make_tensor(
                        name="hoge",
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[],
                        vals=[x],
                    )
                )
            )
            return res

        lv = totensor(lv)
        rv = totensor(rv)

        env.nodes.append(
            helper.make_node(
                optype,
                inputs=[lv.name, rv.name], outputs=[res.name],
            )
        )
        return res
    elif isinstance(nast, gast.Attribute):
        # TODO(satos) 多段attrに対応する

        # code.InteractiveConsole({'nast':nast,'env': env}).interact()
        body = eval_ast(nast.value, env)
        # code.InteractiveConsole({'nast':nast,'env': env, 'body':body}).interact()
        if body == chainer.functions:
            v = getattr(body, nast.attr)
            for f, c in Func2NodeClass:
                if v == f:
                    return c()
            else:
                raise Exception('unknown function', nast.attr)
        else:
            return body[nast.attr]

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
        localenv.vars = copy.deepcopy(env.vars)
        tx = new_tensor()
        localenv.vars[x] = tx
        ty = eval_ast(nast.elt,  localenv)

        # Scan は map の map なので、一旦[x]で包んでかけてしまう

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
            inputs=[xs.name], outputs=[bres.name],
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
        step = nast.slice.step
        lower = nast.slice.lower
        upper = nast.slice.upper
        """
        if not (step is None) and eval_ast(step,env) == -1:
            if lower is None and upper is None:
               # 反転
            else:
                raise Exception("Unimplemented")
        else:
            raise Exception("Unimplemented")
        """

        lower = [] if lower is None else lower
        upper = [] if upper is None else upper

        res = new_tensor()
        env.addnode(
            'Slice',
            inputs=[vs.name], outputs=[res.name],
            starts=lower,
            ends=upper
        )
        return res

    elif isinstance(nast, gast.Name):
        if nast.id == 'print':  # とりあえずの実装なのであとでもっとうまくやる
            return Func(lambda _, __, ___: None)
        elif nast.id in env.vars.keys():
            return env.vars[nast.id]
        elif nast.id in dir(env.module):
            return getattr(env.module, nast.id)
        elif nast.id == env.self_name:
            return env.self
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
    for i, _ in enumerate(molk.forward_args[1:]):  # self 以外
        x = new_tensor(['batch_size%d' % i, 'input_size%d' % i])
        input_tensors.append(x)

    env = Env()
    v = molk.call(input_tensors, [], env)  # keywordsはとりあえず空

    if not get_test_args().quiet:
        print(v)
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

    if not get_test_args().quiet:
        print(graph)
    # exit(0)
    # checker.check_graph(graph)
    # oniku独自のノードを使うとcheckできなくなる...
    mo = helper.make_model(graph)

    # print(mo)
    return mo, input_tensors, output_tensors
