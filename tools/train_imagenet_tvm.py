# A wrapper of train_imagenet with Python TVM support.

import os
import sys

oniku_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(oniku_root, 'python'))
sys.path.append(os.path.join(oniku_root, 'build/tools'))

import oniku_tvm

import train_imagenet_core


def main():
    oniku_tvm.init()
    train_imagenet_core.train_imagenet(
        sys.argv + ['--fuse_operations', '--use_tvm'])


if __name__ == '__main__':
    main()
