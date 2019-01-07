import argparse


_args_cache = None


def get_test_args(args=None):
    global _args_cache
    if _args_cache is not None:
        return _args_cache

    parser = argparse.ArgumentParser(
        description='A test for Python => ONNX compiler')
    parser.add_argument('output', type=str,
                        help='An output ONNX testcase directory.')
    parser.add_argument('--nocheck', dest='check', action='store_false',
                        help='Do not run the check with mxnet.')
    parser.add_argument('--quiet', action='store_true',
                        help='Show less messages.')
    parser.add_argument('--allow-unused-params', action='store_true',
                        help='Allow unused parameters.')
    _args_cache = parser.parse_args(args=args)
    return _args_cache


def dprint(*v):
    if not get_test_args().quiet:
        print(*v)
