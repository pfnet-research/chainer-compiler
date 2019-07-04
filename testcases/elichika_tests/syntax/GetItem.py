# coding: utf-8

import chainer


class GetItemSimple(chainer.Chain):
    def __init__(self):
        super(GetItemSimple, self).__init__()
        self.x = 1

    def forward(self):
        self.y = 2
        return self['x'] * self['y']

class GetItemCustom(chainer.Chain):
    def __init__(self):
        super(GetItemCustom, self).__init__()
        self.x = 1

    def __getitem__(self, name):
        self.x += 1
        return getattr(self, name)

    def forward(self):
        self.y = 2
        return self['x'] * self['y']


class GetItemFunction(chainer.Chain):
    def func(self, n):
        return n

    def forward(self):
        self.x = 3
        return self['func'](self['x'])


class GetItemGetter(chainer.Chain):
    def __init__(self):
        super(GetItemGetter, self).__init__()
        self._x = 1

    @property
    def x(self):
        return self._x

    def forward(self):
        return self['x']

class StrConstantPropagation(chainer.Chain):
    def __init__(self):
        super(StrConstantPropagation, self).__init__()
        self.x1 = 1
        self.x2 = 2
        self.x3 = 3
        self.x121hello1 = 4

    def forward(self):
        ret = 0
        for i in [1, 2, 3]:
            ret += self['x%d' % i]
        ret += self['x%s' % '%d%s' % (121, "%s%d" % ("hello", True))]
        return ret


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np


def main():
    testtools.generate_testcase(GetItemSimple(), [], subname='GetItemSimple')
    # testtools.generate_testcase(GetItemCustom(), [], subname='GetItemCustom')  # Bug in UserDefinedFunctions.
    testtools.generate_testcase(GetItemFunction(), [], subname='GetItemFunction')
    testtools.generate_testcase(GetItemGetter(), [], subname='GetItemGetter')
    testtools.generate_testcase(StrConstantPropagation(), [], subname='StrConstantPropagation')


if __name__ == '__main__':
    main()
