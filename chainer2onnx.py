# coding: utf-8

import ast
import gast
import inspect
import onnx
from onnx import checker
from onnx import helper
from onnx import TensorProto
import os
import sys

from chainer import functions as F
from chainer import links as L

import code
import test_args


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
        # code.InteractiveConsole({'ch': ch}).interact()

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
                inputs=[v.name, sl.W.name] +
                ([] if sl.b is None else [sl.b.name]),
                outputs=[res.name],
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
        # code.InteractiveConsole({'ch': ch}).interact()

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
                inputs=[v.name, sl.scale.name, sl.B.name,
                        sl.mean.name, sl.var.name], outputs=[res.name],
                epsilon=sl.eps,
                momentum=sl.momentum,
                # とりあえずspatialは1で(0でも値が変わらなかったのでよくわからん)
            )
        )
        return res

    def init_tensors(sl):
        return [sl.scale, sl.B, sl.mean, sl.var]


class Link_NstepLSTM(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + '_' + ch.name
        #code.InteractiveConsole({'ch': ch}).interact()

        cs = list(ch.children())
        sl.n_layers = ch.n_layers
        if not(cs[0].w0 is None):
            sl.n_in = cs[0].w0.shape[1]
        else:
            sl.n_in = None

        sl.out_size = ch.out_size
        sl.dropout = ch.dropout

        class step(object):
            def __init__(sl):
                pass

        sl.ws = [step() for _ in range(sl.n_layers)]
        for i in range(sl.n_layers):
            sl.ws[i].W = helper.make_tensor_value_info(
                sl.name + ('_%d_ws0' % i), TensorProto.FLOAT, ["TODO"])
            # これ多分うまいこと変換しないといけない
            # chainer : at  ct
            #   onnx  : ct  Ct
            # (chainerのws[0],ws[2],ws[1],ws[3]から連結させたりする)
            sl.ws[i].R = helper.make_tensor_value_info(
                sl.name + ('_%d_ws1' % i), TensorProto.FLOAT, ["TODO"])
            # (chainerのws[4],ws[6],ws[5],ws[7]から連結させたりする)
            sl.ws[i].B = helper.make_tensor_value_info(
                sl.name + ('_%d_bss' % i), TensorProto.FLOAT, ["TODO"])
            # (chainerのbs[0,2,1,3,4,6,5,7]から連結させたりする)

    def call(sl, args, _, env):
        # とりあえずnstep を 1step ずつに分解する
        # print(sl.name,args)
        #assert(len(args) == 1)
        assert(args[0] is None and args[1] is None)
        v = args[2]

        hs = []
        cs = []

        for i in range(sl.n_layers):

            h = new_tensor(['unknown', 'unknown', 'unknown'])
            c = new_tensor(['unknown', 'unknown', 'unknown'])
            ys = new_tensor(['unknown', 'unknown', 'unknown'])

            env.nodes.append(
                helper.make_node(
                    "LSTM",
                    inputs=[v.name, sl.ws[i].W.name,
                            sl.ws[i].R.name, sl.ws[i].B.name],
                    outputs=[ys.name, h.name, c.name],
                    hidden_size=sl.out_size
                )
            )

            hs.append(h.name)
            cs.append(c.name)
            v = ys
        print(hs)
        print(cs)
        ths = new_tensor(['unknown', 'unknown', 'unknown'])
        tcs = new_tensor(['unknown', 'unknown', 'unknown'])
        env.nodes.append(
            helper.make_node(
                "Concat",
                inputs=hs, outputs=[ths.name],
                axis=0,
            )
        )
        env.nodes.append(
            helper.make_node(
                "Concat",
                inputs=cs, outputs=[tcs.name],
                axis=0,
            )
        )
        tys = v
        return ths, tcs, tys

    def init_tensors(sl):
        return sum([[sl.ws[i].W, sl.ws[i].B, sl.ws[i].R] for i in range(sl.n_layers)], [])


class User_Defined_Link(object):
    def __init__(sl, ch, parentname):
        sl.name = parentname + ('' if ch.name is None else '_' + ch.name)
        # code.InteractiveConsole({'ch': ch}).interact()

        src = clip_head(inspect.getsource(ch.forward))
        if not test_args.get_test_args().quiet:
            print(src)
        sl.ast = gast.ast_to_gast(ast.parse(src)).body[0]
        assert(isinstance(sl.ast, gast.gast.FunctionDef))

        sl.links = initLinks(ch, sl.name)
        sl.module = sys.modules[ch.__module__]
        args = list(map(lambda x: x.id, sl.ast.args.args))
        sl.self_name = args[0]
        sl.arg_name = args[1:]

    def call(sl, args, _, env):
        # 自身のforwardを呼ぶ
        # print('calling',sl.name,'with',args)
        # code.InteractiveConsole({'nast':sl.ast}).interact()

        loenv = env.localenv()

        loenv.self_name = sl.self_name
        assert(len(sl.arg_name) == len(args))
        loenv.vars = dict(zip(sl.arg_name, args))

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
            # pads = [dx, dy, strides[0]+dx-1, strides[1]+dy-1]
            # 多めに足しておくとうまいこといくはず (この時点で入力大きさが不明なので)
            # mxnetだと足しておくとよかったが、
            # Onikuだとそうではないっぽい？

            # (size + pad) % stride = ksize % stride
            # を仮定してよい？

            pads = [dx, dy, dx, dy]
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


class Function_SoftmaxClossEntropy(object):
    def call(sl, args, keywords, env):
        assert(len(args) == 2)
        v, w = args[0], args[1]
        res = new_tensor(get_dims(v))
        env.nodes.append(
            helper.make_node(
                "OnikuxSoftmaxCrossEntropy", inputs=[v.name, w.name], outputs=[res.name]
            )
        )
        return res


class Func(object):
    def __init__(sl, f):
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
    (F.average_pooling_2d, Function_AveragePool2d),
    (F.softmax_cross_entropy, Function_SoftmaxClossEntropy)
]

Link2NodeClass = [
    (L.Linear, Link_Linear),
    (L.Convolution2D, Link_Convolution2D),
    (L.BatchNormalization, Link_BatchNormalization),
    (L.NStepLSTM, Link_NstepLSTM)
]


def eval_ast(nast, env):
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
            return lv * rv

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
        na = nast.value.id
        if na == env.self_name:  # .selfのとき
            if nast.attr == 'children':
                # code.InteractiveConsole({'nast':nast,'env': env}).interact()
                # これでよさそう?(順番があってるのかあやしい)
                return Func(lambda _, __, ___: env.links.values())
            else:
                return env.links[nast.attr]
        elif na in dir(env.module):
            v = getattr(getattr(env.module, na), nast.attr)
            for f, c in Func2NodeClass:
                if v == f:
                    return c()
            else:
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
    elif isinstance(nast, gast.List):
        return list(map(lambda x: eval_ast(x, env), nast.elts))
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


def chainer2onnx(model, forward):
    # return helper.make_graph([],'dummy',[],[])

    # code.InteractiveConsole({'mo': model}).interact()
    molk = User_Defined_Link(model, '')

    input_tensors = []
    for i, _ in enumerate(molk.arg_name):
        x = new_tensor(['batch_size%d' % i, 'input_size%d' % i])
        input_tensors.append(x)

    env = Env()
    v = molk.call(input_tensors, [], env)  # keywordsはとりあえず空

    if not test_args.get_test_args().quiet:
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

    if not test_args.get_test_args().quiet:
        print(graph)
    # exit(0)
    # checker.check_graph(graph)
    # oniku独自のノードを使うとcheckできなくなる...
    mo = helper.make_model(graph)

    # print(mo)
    return mo, input_tensors, output_tensors
