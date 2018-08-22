# coding: utf-8

import ast
import gast
import inspect
from onnx import checker
from onnx import helper
from onnx import TensorProto
import os
import sys

from chainer import functions as F
from chainer import links as L

import code


def new_tensor(dims):
    tn = new_tensor.cnt
    new_tensor.cnt += 1
    return helper.make_tensor_value_info(
        'T' + str(tn), TensorProto.FLOAT, dims)


new_tensor.cnt = 0


def get_dims(tensor):
    dims = tensor.type.tensor_type.shape.dim
    return list(map(lambda x: x.dim_value, dims))


class Link_Linear(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        sl.n_out = ch.b.shape[0]
        if not(ch.W.data is None):
            sl.n_in = ch.W.shape[1]
        else:
            sl.n_in = None

        sl.W = helper.make_tensor_value_info(
            sl.name + '_W', TensorProto.FLOAT,
            [sl.n_out, ('input_size' if (sl.n_in is None) else sl.n_in)])
        sl.b = helper.make_tensor_value_info(
            sl.name + '_b', TensorProto.FLOAT, [sl.n_out])

    def call(sl, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor([sl.n_out])
        env.nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[v.name, sl.W.name, sl.b.name], outputs=[res.name],
                transA=0, transB=1
            )
        )
        return res

    def init_tensors(sl):
        return [sl.W, sl.b]


def size2d(v):
    if isinstance(v, tuple):
        return list(v)
    elif isinstance(v, int):
        return [v, v]
    else:
        raise Exception('size should be tuple or int, but got ', v)


class Link_Convolution2D(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        #code.InteractiveConsole({'ch': ch}).interact()

        sl.ksize = size2d(ch.ksize)
        sl.stride = size2d(ch.stride)
        ps = size2d(ch.pad)
        sl.pads = ps + ps
        
        if not (ch.b is None):
            # nobias = True の場合
            sl.M = ch.b.shape[0]
            sl.b = helper.make_tensor_value_info(
                sl.name + '_b', TensorProto.FLOAT, [sl.M])
        else:
            sl.M = "TODO"
            sl.b = None

        sl.W = helper.make_tensor_value_info(
            sl.name + '_W', TensorProto.FLOAT,
            [sl.M, 'channel_size'] + sl.ksize)


    def call(sl, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Conv",
                inputs=[v.name, sl.W.name] + ([] if sl.b is None else [sl.b.name]), outputs=[res.name],
                kernel_shape=sl.ksize,
                pads=sl.pads,
                strides=sl.stride
            )
        )
        return res

    def init_tensors(sl):
        return [sl.W] + ([] if sl.b is None else [sl.b])


class Link_BatchNormalization(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        #code.InteractiveConsole({'ch': ch}).interact()
        
        sl.n_out = ch.beta.shape[0]

        sl.scale = helper.make_tensor_value_info(
            sl.name + '_gamma', TensorProto.FLOAT, [sl.n_out])
        sl.B = helper.make_tensor_value_info(
            sl.name + '_beta', TensorProto.FLOAT, [sl.n_out])
        sl.mean = helper.make_tensor_value_info(
            sl.name + '_avg_mean', TensorProto.FLOAT, [sl.n_out])
        sl.var = helper.make_tensor_value_info(
            sl.name + '_avg_var', TensorProto.FLOAT, [sl.n_out])

        sl.eps = ch.eps
        sl.momentum = ch.decay 
        
    def call(sl, args, _, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "BatchNormalization",
                inputs=[v.name, sl.scale.name, sl.B.name, sl.mean.name, sl.var.name], outputs=[res.name],
                epsilon=sl.eps,
                momentum=sl.momentum,
                # とりあえずspatialは1で(0でも値が変わらなかったのでよくわからん)
            )
        )
        return res

    def init_tensors(sl):
        return [sl.scale, sl.B, sl.mean, sl.var]


class User_Defined_Link(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        # code.InteractiveConsole({'ch': ch}).interact()

        src = clip_head(inspect.getsource(ch.forward))
        print(src)
        sl.ast = gast.ast_to_gast(ast.parse(src)).body[0]
        assert(isinstance(sl.ast, gast.gast.FunctionDef))

        sl.links = initLinks(ch, sl.name)
        sl.module = sys.modules[ch.__module__]

    def call(sl, args, _, env):
        # 自身のforwardを呼ぶ
        # print('calling',sl.name,'with',args)
        # code.InteractiveConsole({'nast':sl.ast}).interact()

        loenv = env.localenv()

        fnargs = list(map(lambda x: x.id, sl.ast.args.args))
        loenv.self_name = fnargs[0]
        assert(len(fnargs) == len(args)+1)
        loenv.vars = dict(zip(fnargs[1:], args))

        loenv.links = sl.links
        loenv.module = sl.module
        print('name', sl.name, 'modules', sl.module)
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


class Function_Relu(object):
    def call(sl, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(get_dims(v))
        env.nodes.append(
            helper.make_node(
                "Relu", inputs=[v.name], outputs=[res.name]
            )
        )
        return res


class Function_Pool2d_Util(object):
    def __init__(sl, pooltype):
        sl.pooltype = pooltype

    def call(sl, args, keywords, env):
        assert(len(args) == 2)
        v = args[0]
        res = new_tensor(['TODO'])
        ksize = args[1]
        strides = size2d(keywords.get('stride', ksize))
        # chainer のsize参考
        # https://github.com/chainer/chainer/blob/v4.3.1/chainer/utils/conv.py#L7

        # paddingについて、Chainerの cover_all=Falseと
        # onnx の pads=0 が一致する
        # ので、 cover_all=True(デフォルト)なら
        # padsを入れる必要あり
        pads = [0, 0, 0, 0]
        if keywords.get('cover_all', True):
            dx, dy = 0, 0
            if 'pad' in keywords.keys():
                dy, dx = size2d(keywords['pad'])
            pads = [dx, dy, strides[0]+dx-1, strides[1]+dy-1]
            # 多めに足しておくとうまいこといくはず (この時点で入力大きさが不明なので)
        else:
            raise Exception("unimplemented cover_all=False in maxpool2d")

        env.nodes.append(
            helper.make_node(
                sl.pooltype, inputs=[v.name], outputs=[res.name],
                kernel_shape=size2d(ksize),
                strides=strides,
                pads=pads
            )
        )
        return res


class Function_MaxPool2d(object):
    def __init__(sl):
        sl.call = Function_Pool2d_Util('MaxPool').call


class Function_AveragePool2d(object):
    def __init__(sl):
        sl.call = Function_Pool2d_Util('AveragePool').call


class Function_LocalRespNorm(object):
    def call(sl, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['TODO'])
        n = keywords.get('n', 5)
        alpha = keywords.get('alpha', 0.0001)
        env.nodes.append(
            helper.make_node(
                "LRN", inputs=[v.name], outputs=[res.name],
                size=n,
                bias=keywords.get('k', 2.0),
                alpha=alpha * n,  # chainerとonnx(mxnet)で一致しない
                beta=keywords.get('beta', 0.75)
            )
        )
        return res


class Function_Dropout(object):
    def call(sl, args, keywords, env):  # たぶん実際には実装できない
        if len(args) == 1:
            pass
        elif len(args) == 2:
            keywords['ratio'] = args[1]
        else:
            raise Exception("invalid length")

        v = args[0]
        res = new_tensor(['TODO'])
        env.nodes.append(
            helper.make_node(
                "Dropout", inputs=[v.name], outputs=[res.name],
                ratio=keywords.get('ratio', 0.5),
            )
        )
        return res


class Function_Concat(object):
    def call(sl, args, keywords, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(['TODO'])
        # print(list(v))
        env.nodes.append(
            helper.make_node(
                "Concat",
                inputs=list(map(lambda x: x.name, v)), outputs=[res.name],
                axis=keywords.get('axis', 1),
            )
        )
        return res


class Func(object):
    def __init__(sl,f):
        sl.call = f

def clip_head(s):
    s = s.split('\n')
    # print(s)
    hs = os.path.commonprefix(list(filter(lambda x: x != '', s)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    s = map(lambda x: x[ls:], s)
    return '\n'.join(s)


class Env(object):
    def __init__(sl):
        sl.vars = {}
        sl.nodes = []

    def localenv(sl):
        res = Env()
        res.nodes = sl.nodes  # こっちはglobalに共通でないといけない
        return res


class ValueReturn(Exception):
    def __init__(sl, value):
        sl.value = value

Func2NodeClass = [
    (F.relu, Function_Relu),
    (F.max_pooling_2d, Function_MaxPool2d),
    (F.local_response_normalization, Function_LocalRespNorm),
    (F.dropout, Function_Dropout),
    (F.concat, Function_Concat),
    (F.average_pooling_2d, Function_AveragePool2d)
]


def eval_ast(nast, env):
    if isinstance(nast, list):
        # 逐次実行
        for s in nast:
            eval_ast(s, env)
    elif isinstance(nast,gast.For):
        # とりあえず実際にfor文を回す
        tg = nast.target.id
        for v in eval_ast(nast.iter,env):
            env.vars[tg] = v
            eval_ast(nast.body,env)
        env.vars.pop(tg)

    elif isinstance(nast, gast.Assign):
        value = eval_ast(nast.value, env)
        targs = nast.targets
        # とりあえずタプル代入はなし
        assert(len(targs) == 1)
        env.vars[targs[0].id] = value
    elif isinstance(nast, gast.Call):
        fn = eval_ast(nast.func, env)
        args = list(map(lambda x: eval_ast(x, env), nast.args))
        keywords = dict(
            map(lambda x: (x.arg, eval_ast(x.value, env)), nast.keywords))
        return fn.call(args, keywords, env)
    elif isinstance(nast, gast.BinOp):
        lv = eval_ast(nast.left, env)
        rv = eval_ast(nast.right, env)
        res = new_tensor(['TODO'])
        if isinstance(nast.op, gast.Add):    
            optype = "Add"
        else:
            raise Exception('unknown operator',nast.op)
        
        env.nodes.append(
            helper.make_node(
                optype,
                inputs=[lv.name,rv.name], outputs=[res.name],
            )
        )
        return res
    elif isinstance(nast, gast.Attribute):
        na = nast.value.id
        if na == env.self_name:  # .selfのとき
            if nast.attr == 'children':
                #code.InteractiveConsole({'nast': nast, 'env': env}).interact()
                return Func(lambda _,__,___: env.links.values()) #これでよさそう?(順番があってるのかあやしい)
            else:
                return env.links[nast.attr]
        elif na in dir(env.module):
            v = getattr(getattr(env.module, na), nast.attr)
            for f, c in Func2NodeClass:
                if v == f:
                    return c()

            raise Exception('unknown function', nast.attr)
        else:
            raise Exception('unknown attribute', nast.value.id, '.', nast.attr)

    elif isinstance(nast, gast.Name):
        if nast.id == 'print':  # とりあえずの実装なのであとでもっとうまくやる
            return Func(lambda _, __, ___: None)
        else:
            return env.vars[nast.id]
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
    elif isinstance(nast, gast.Return):
        raise ValueReturn(eval_ast(nast.value, env))
    else:
        print('unknown ast')
        code.InteractiveConsole({'nast': nast, 'env': env}).interact()
        raise Exception('unknown ast', nast)


def initLinks(model, parentname):
    links = {}
    for ch in model.children():
        if ch.__class__.__module__[:13] == 'chainer.links':
            if isinstance(ch, L.Linear):
                links[ch.name] = Link_Linear(ch, parentname)
            elif isinstance(ch, L.Convolution2D):
                links[ch.name] = Link_Convolution2D(ch, parentname)
            elif isinstance(ch, L.BatchNormalization):
                links[ch.name] = Link_BatchNormalization(ch, parentname)
            else:
                print('unknown chainer link')
                code.InteractiveConsole({'lk': ch}).interact()
                raise Exception('unknown link', ch)
        else:
            # User Defined link
            links[ch.name] = User_Defined_Link(ch, parentname)

    return links


def chainer2onnx(model, forward):
    # return helper.make_graph([],'dummy',[],[])

    links = initLinks(model, '')
    # ここまでinit,以下forward

    src = clip_head(inspect.getsource(forward))
    print(src)
    tast = gast.ast_to_gast(ast.parse(src))

    tast = tast.body[0]
    assert(isinstance(tast, gast.gast.FunctionDef))

    env = Env()
    env.links = links
    env.module = sys.modules[model.__module__]

    args = list(map(lambda x: x.id, tast.args.args))
    env.self_name = args[0]

    args = args[1:]
    # code.InteractiveConsole({'tast':tast}).interact()
    assert(len(args) == 1)  # とりあえず1入力
    input_tensors = []
    for v in args:
        env.vars[v] = new_tensor(['batch_size', 'input_size'])
        input_tensors.append(env.vars[v])
    try:
        eval_ast(tast.body, env)
        raise Exception('return not found')
    except ValueReturn as v:
        if isinstance(v.value, tuple):
            output_tensors = list(v.value)  # ばらしてみる
        else:
            output_tensors = [v.value]  # とりあえず1tensor

    for lk in links.values():
        # print(lk,a)
        input_tensors += lk.init_tensors()

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

    print(graph)
    # exit(0)
    checker.check_graph(graph)
    mo = helper.make_model(graph)

    # print(mo)
    return mo
