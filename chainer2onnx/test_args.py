import argparse


_args_cache = None


def get_test_args():
    global _args_cache
    if _args_cache is not None:
        return _args_cache

    parser = argparse.ArgumentParser(
        description='A test for Python => ONNX compiler')
    parser.add_argument('--raw_output', type=str, default='raw_MLP.onnx',
                        help='An output ONNX graph without parameters.')
    parser.add_argument('--output', type=str, default='initialized_MLP.onnx',
                        help='An output ONNX graph with parameters.')
    parser.add_argument('--test_data_dir', type=str, default='',
                        help='If specified, output test inputs/outputs.')
    parser.add_argument('--nocheck', dest='check', action='store_false',
                        help='Do not run the check with mxnet.')
    parser.add_argument('--quiet', action='store_true',
                        help='Show less messages.')
    _args_cache = parser.parse_args()
    return _args_cache
