# A wrapper of run_onnx with Python TVM support.

import os
import sys

oniku_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(oniku_root, 'python'))
sys.path.append(os.path.join(oniku_root, 'build/tools'))

import oniku_tvm

import run_onnx_core


def main():
    oniku_tvm.init()
    run_onnx_core.run_onnx(sys.argv + ['--fuse_operations', '--use_tvm'])


if __name__ == '__main__':
    main()
