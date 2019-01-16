#!python3
#
# A wrapper of train_imagenet with Python TVM support.

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'python'))
sys.path.append(os.path.join(project_root, 'build/tools'))

import oniku_tvm

import train_imagenet_core


def _get_args():
    args = list(sys.argv)
    if '--tvm' in args:
        args.remove('--tvm')
        args.extend(['--fuse_operations', '--use_tvm'])
    return args


def main():
    oniku_tvm.init()
    train_imagenet_core.train_imagenet(_get_args())


if __name__ == '__main__':
    main()
