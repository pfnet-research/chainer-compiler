# A wrapper of run_onnx with Python TVM support.

import os
import sys

import topi
import tvm

oniku_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(oniku_root, 'build/tools'))

import run_onnx_core


def main():
    run_onnx_core.run_onnx(sys.argv + ['--fuse_operations', '--use_tvm'])


if __name__ == '__main__':
    main()
