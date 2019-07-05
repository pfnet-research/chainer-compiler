import argparse
import glob
import logging
import os
import sys

import cupy
import nnvm
import nnvm.compiler
import numpy as np
import onnx
import onnx.numpy_helper
import tvm

from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime

import run_onnx_util


def load_test_data(data_dir, input_names, output_names):
    inout_values = []
    for kind, names in [('input', input_names), ('output', output_names)]:
        names = list(names)
        values = []
        for pb in sorted(glob.glob(os.path.join(data_dir, '%s_*.pb' % kind))):
            with open(pb, 'rb') as f:
                tensor = onnx.TensorProto()
                tensor.ParseFromString(f.read())
            if tensor.name in names:
                name = tensor.name
                names.remove(name)
            else:
                name = names.pop(0)
            values.append((name, onnx.numpy_helper.to_array(tensor)))
        inout_values.append(values)
    return tuple(inout_values)


def compile(symbol, target, input_names, inputs, params,
            opt_level, autotvm_log):
    shape_dict = {}
    dtype_dict = {}
    for name, value in zip(input_names, inputs.values()):
        shape_dict[name] = value.shape
        dtype_dict[name] = value.dtype
    for name, value in params.items():
        shape_dict[name] = value.shape
        dtype_dict[name] = value.dtype
    with nnvm.compiler.build_config(opt_level=opt_level):
        if autotvm_log:
            with tvm.autotvm.apply_history_best(autotvm_log):
                graph, lib, params = nnvm.compiler.build(symbol, target,
                                                         shape=shape_dict,
                                                         dtype=dtype_dict,
                                                         params=params)
        else:
            graph, lib, params = nnvm.compiler.build(symbol, target,
                                                     shape=shape_dict,
                                                     dtype=dtype_dict,
                                                     params=params)
    return graph, lib, params


def run(args):
    onnx_model = onnx.load_model(os.path.join(args.test_dir, 'model.onnx'))
    symbol, params = nnvm.frontend.from_onnx(onnx_model)
    input_names = symbol.list_input_names()
    output_names = symbol.list_output_names()

    test_data_dir = os.path.join(args.test_dir, 'test_data_set_0')
    inputs, outputs = load_test_data(test_data_dir, input_names, output_names)
    inputs = dict(inputs)

    # assert len(input_names) == len(inputs) + len(params)
    # assert len(output_names) == len(outputs)

    graph, lib, params = compile(
        symbol, args.target, input_names, inputs, params,
        args.opt_level, args.autotvm_log)

    if args.dump_nnvm:
        print(graph.ir())
        print(graph.json())

    ctx = tvm.gpu()

    # Prepare inputs.
    tvm_inputs = {}
    for name, value in inputs.items():
        tvm_inputs[name] = tvm.nd.array(value, ctx=ctx)
    for name, value in params.items():
        tvm_inputs[name] = tvm.nd.array(value, ctx=ctx)

    graph_module = None
    if args.debug:
        try:
            graph_module = debug_runtime.create(graph, lib, ctx)
        except:
            print('debug_runtime is disabled. '
                  'Set USE_GRAPH_RUNTIME_DEBUG=ON and rebuild TVM')
    if graph_module is None:
        graph_module = graph_runtime.create(graph, lib, ctx)

    graph_module.set_input(**tvm_inputs)

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
    parser.add_argument('--dump_nnvm', action='store_true')
    parser.add_argument('--target', type=str, default='cuda')
    parser.add_argument('--debug', '-g', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    parser.add_argument('--opt_level', '-O', type=int, default=3)
    parser.add_argument('--autotvm_log', type=str)
    return parser.parse_args(args=args)


def main():
    args = get_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == '__main__':
    main()
