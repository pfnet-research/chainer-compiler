import argparse
import logging
import os

import cupy
import numpy as np
import onnx
import onnx.numpy_helper
import tvm

from tvm.contrib.debugger import debug_runtime
from tvm.contrib import graph_runtime

import run_onnx_util


def set_inputs(graph_module, ctx, inputs, params):
    tvm_inputs = {}
    for name, value in inputs.items():
        tvm_inputs[name] = tvm.nd.array(value, ctx=ctx)
    for name, value in params.items():
        tvm_inputs[name] = tvm.nd.array(value, ctx=ctx)
    graph_module.set_input(**tvm_inputs)


def compile_nnvm(symbol, target, input_names, inputs, params,
                 opt_level):
    import nnvm
    import nnvm.compiler

    shape_dict = {}
    dtype_dict = {}
    for name, value in zip(input_names, inputs.values()):
        shape_dict[name] = value.shape
        dtype_dict[name] = value.dtype
    for name, value in params.items():
        shape_dict[name] = value.shape
        dtype_dict[name] = value.dtype
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(symbol, target,
                                                 shape=shape_dict,
                                                 dtype=dtype_dict,
                                                 params=params)
    return graph, lib, params


def create_graph_module(args, graph, lib, ctx):
    graph_module = None
    if args.debug:
        try:
            graph_module = debug_runtime.create(graph, lib, ctx)
        except Exception:
            print('debug_runtime is disabled. '
                  'Set USE_GRAPH_RUNTIME_DEBUG=ON and rebuild TVM')
    if graph_module is None:
        graph_module = graph_runtime.create(graph, lib, ctx)

    return graph_module


def build_graph_nnvm(args, ctx, onnx_model, inputs, input_names):
    import nnvm
    symbol, params = nnvm.frontend.from_onnx(onnx_model)
    # input_names = symbol.list_input_names()
    # output_names = symbol.list_output_names()

    # assert len(input_names) == len(inputs) + len(params)
    # assert len(output_names) == len(outputs)

    graph, lib, params = compile_nnvm(
        symbol, args.target, input_names, inputs, params,
        args.opt_level)

    if args.dump_frontend:
        print(graph.ir())
        print(graph.json())

    graph_module = create_graph_module(args, graph, lib, ctx)
    set_inputs(graph_module, ctx, inputs, params)

    return graph_module


def build_graph_relay(args, ctx, onnx_model, inputs, input_names):
    import tvm.relay as relay

    shape_dict = {}
    dtype_dict = {}
    for name, value in zip(input_names, inputs.values()):
        shape_dict[name] = value.shape
        dtype_dict[name] = value.dtype

    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)
    with relay.build_config(opt_level=args.opt_level):
        graph, lib, params = relay.build(mod, args.target, params=params)

        if args.dump_frontend:
            print(graph)

    graph_module = create_graph_module(args, graph, lib, ctx)
    set_inputs(graph_module, ctx, inputs, params)

    return graph_module


def run(args):
    onnx_model = onnx.load_model(run_onnx_util.onnx_model_file(args.test_dir, args.model_file))
    ctx = tvm.gpu()

    input_names, output_names = run_onnx_util.onnx_input_output_names(
        os.path.join(args.test_dir, args.model_file))

    test_data_dir = os.path.join(args.test_dir, 'test_data_set_0')
    inputs, outputs = run_onnx_util.load_test_data(
        test_data_dir, input_names, output_names)

    inputs = dict(inputs)
    graph_module = None
    if args.frontend == 'nnvm':
        graph_module = build_graph_nnvm(args, ctx, onnx_model, inputs, input_names)
    elif args.frontend == 'relay':
        graph_module = build_graph_relay(args, ctx, onnx_model, inputs, input_names)
    else:
        raise RuntimeError('Invalid frontend: {}'.format(args.frontend))

    graph_module.run()

    for i, (name, expected) in enumerate(outputs):
        tvm_output = tvm.nd.empty(expected.shape, expected.dtype, ctx=ctx)
        actual = graph_module.get_output(i, tvm_output).asnumpy()
        np.testing.assert_allclose(expected, actual,
                                   rtol=1e-3, atol=1e-4), name
        print('%s: OK' % name)
    print('ALL OK')

    def compute():
        graph_module.run()
        cupy.cuda.device.Device().synchronize()

    return run_onnx_util.run_benchmark(compute, args.iterations)


def get_args(args=None):
    parser = argparse.ArgumentParser(description='Run ONNX by TVM')
    parser.add_argument('test_dir')
    parser.add_argument('--frontend', type=str, default='relay')
    parser.add_argument('--dump_frontend', action='store_true')
    parser.add_argument('--target', type=str, default='cuda')
    parser.add_argument('--debug', '-g', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    parser.add_argument('--opt_level', '-O', type=int, default=3)
    parser.add_argument('--autotvm_log', type=str)
    parser.add_argument('--model_file', default=None)
    return parser.parse_args(args=args)


def main():
    args = get_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.autotvm_log:
        with tvm.autotvm.apply_history_best(args.autotvm_log):
            run(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
