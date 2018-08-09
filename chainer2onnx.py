# coding: utf-8

import ast
import gast
import inspect
from onnx import checker
from onnx import helper
from onnx import TensorProto
import os


from chainer import links as L


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
    def __init__(sl, ch):
        sl.name = ch.name
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

    def call(sl, args, env):
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


class Function_Relu(object):
    def call(sl, args, env):
        assert(len(args) == 1)
        v = args[0]
        res = new_tensor(get_dims(v))
        env.nodes.append(
            helper.make_node(
                "Relu", inputs=[v.name], outputs=[res.name]
            )
        )
        return res


def clip_head(s):
    s = s.split('\n')
    # print(s)
    hs = os.path.commonprefix(list(filter(lambda x: x != '', s)))
    # print('hs',list(map(ord,hs)))
    ls = len(hs)
    s = map(lambda x: x[ls:], s)
    return '\n'.join(s)


class Env(object):
    def __init__(sl, links):
        sl.links = links
        sl.vars = {}
        sl.nodes = []


class ValueReturn(Exception):
    def __init__(sl, value):
        sl.value = value


def eval_ast(nast, env):
    if isinstance(nast, list):
        # 逐次実行
        for s in nast:
            eval_ast(s, env)
    elif isinstance(nast, gast.Assign):
        value = eval_ast(nast.value, env)
        targs = nast.targets
        # とりあえずタプル代入はなし
        assert(len(targs) == 1)
        env.vars[targs[0].id] = value
    elif isinstance(nast, gast.Call):
        fn = eval_ast(nast.func, env)
        args = list(map(lambda x: eval_ast(x, env), nast.args))
        return fn.call(args, env)
    elif isinstance(nast, gast.Attribute):
        if nast.value.id == env.self_name:  # .selfのとき
            return env.links[nast.attr]
        # elif
        # これfunction を F とimport してるかどうかって、関数側からわかんないですよね
        elif nast.value.id == 'F':
            if nast.attr == 'relu':
                return Function_Relu()
            else:
                raise Exception('unknown function', nast.attr)
        else:
            raise Exception('unknown attribute', nast.value.id, '.', nast.attr)

    elif isinstance(nast, gast.Name):
        return env.vars[nast.id]
    elif isinstance(nast, gast.Return):
        raise ValueReturn(eval_ast(nast.value, env))
    else:
        raise Exception('unknown ast', nast)


def chainer2onnx(model, forward):
    links = {}
    for ch in model.children():
        if isinstance(ch, L.Linear):
            # print(ch)
            links[ch.name] = Link_Linear(ch)
        else:
            raise Exception('unknown link', ch)

    src = clip_head(inspect.getsource(forward))
    print(src)
    tast = gast.ast_to_gast(ast.parse(src))

    tast = tast.body[0]
    assert(isinstance(tast, gast.gast.FunctionDef))

    env = Env(links)
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

    checker.check_graph(graph)
    mo = helper.make_model(graph)

    print(mo)
    return mo
