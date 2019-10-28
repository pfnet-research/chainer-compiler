import argparse
import logging
import os
import sys

import chainer
import cupy
import numpy as np
import onnx
import onnx.numpy_helper
import tensorrt

import run_onnx_util


def to_gpu(arrays):
    return [cupy.array(a) for a in arrays]


def to_cpu(arrays):
    out = []
    for a in arrays:
        if isinstance(a, chainer.Variable):
            a = a.array
        out.append(chainer.cuda.to_cpu(a))
    return out


def run(args):
    onnx_filename = run_onnx_util.onnx_model_file(args.test_dir, args.model_file)
    input_names, output_names = run_onnx_util.onnx_input_output_names(
        onnx_filename)
    test_data_dir = os.path.join(args.test_dir, 'test_data_set_0')
    inputs, outputs = run_onnx_util.load_test_data(
        test_data_dir, input_names, output_names)

    with open(onnx_filename, 'rb') as f:
        onnx_proto = f.read()

    if args.debug:
        logger = tensorrt.Logger(tensorrt.Logger.Severity.INFO)
    else:
        logger = tensorrt.Logger()
    builder = tensorrt.Builder(logger)
    if args.fp16_mode:
        builder.fp16_mode = True
    # TODO(hamaji): Infer batch_size from inputs.
    builder.max_batch_size = args.batch_size
    network = builder.create_network()
    parser = tensorrt.OnnxParser(network, logger)
    if not parser.parse(onnx_proto):
        for i in range(parser.num_errors):
             sys.stderr.write('ONNX import failure: %s\n' % parser.get_error(i))
             raise RuntimeError('ONNX import failed')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()

    assert len(inputs) + len(outputs) == engine.num_bindings
    for i, (_, input) in enumerate(inputs):
        assert args.batch_size == input.shape[0]
        assert input.shape[1:] == engine.get_binding_shape(i)
    for i, (_, output) in enumerate(outputs):
        assert args.batch_size == output.shape[0]
        i += len(inputs)
        assert output.shape[1:] == engine.get_binding_shape(i)

    inputs = [v for n, v in inputs]
    outputs = [v for n, v in outputs]
    gpu_inputs = to_gpu(inputs)
    gpu_outputs = []
    for output in outputs:
        gpu_outputs.append(cupy.zeros_like(cupy.array(output)))
    bindings = [a.data.ptr for a in gpu_inputs]
    bindings += [a.data.ptr for a in gpu_outputs]

    context.execute(args.batch_size, bindings)

    actual_outputs = to_cpu(gpu_outputs)

    for i, (name, expected, actual) in enumerate(
            zip(output_names, outputs, actual_outputs)):
        np.testing.assert_allclose(expected, actual,
                                   rtol=args.rtol, atol=args.atol), name
        print('%s: OK' % name)
    print('ALL OK')

    def compute():
        context.execute(args.batch_size, bindings)
        cupy.cuda.device.Device().synchronize()

    return run_onnx_util.run_benchmark(compute, args.iterations)


def get_args(args=None):
    parser = argparse.ArgumentParser(description='Run ONNX by TensorRT')
    parser.add_argument('test_dir')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--debug', '-g', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    parser.add_argument('--fp16_mode', action='store_true')
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--model_file', default=None)
    return parser.parse_args(args=args)


def main():
    args = get_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == '__main__':
    main()
