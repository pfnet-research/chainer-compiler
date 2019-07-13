# coding: utf-8

import chainer

class Print(chainer.Chain):
    def forward(self, x):
        print('test', x)
        return 0

class PrintWithConstantPropagation(chainer.Chain):
    def forward(self):
        print('foo', 'test %d' % 3)
        return 0

# ======================================


from chainer_compiler.elichika import testtools
import numpy as np

def main():
    testtools.generate_testcase(Print, [10])
    testtools.generate_testcase(PrintWithConstantPropagation, [],
                                subname='const_prop')
if __name__ == '__main__':
    main()
