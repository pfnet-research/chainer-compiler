import argparse
import logging
import os
import sys

import numpy as np
import onnx
import onnx_tf
import onnx.numpy_helper
import tensorflow as tf

import run_onnx_util


def run(args):
    onnx_filename = run_onnx_util.onnx_model_file(args.test_dir, args.model_file)
    input_names, output_names = run_onnx_util.onnx_input_output_names(
        onnx_filename)
    test_data_dir = os.path.join(args.test_dir, 'test_data_set_0')
    inputs, outputs = run_onnx_util.load_test_data(
        test_data_dir, input_names, output_names)

    model = onnx.load(onnx_filename)
    tf_model = onnx_tf.backend.prepare(model)

    inputs = dict(inputs)
    outputs = dict(outputs)
    actual_outputs = tf_model.run(inputs)

    for name in output_names:
        expected = outputs[name]
        actual = actual_outputs[name]
        np.testing.assert_allclose(expected, actual,
                                   rtol=1e-3, atol=1e-4), name
        print('%s: OK' % name)
    print('ALL OK')

    def compute():
        tf_model.run(inputs)

    return run_onnx_util.run_benchmark(compute, args.iterations)


def get_args(args=None):
    parser = argparse.ArgumentParser(description='Run ONNX by nGraph')
    parser.add_argument('test_dir')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    parser.add_argument('--model_file', default=None)
    return parser.parse_args(args=args)


def main():
    args = get_args()
    run(args)


if __name__ == '__main__':
    main()
