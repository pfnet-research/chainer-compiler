# coding: utf-8

import chainer


class HasItemSimple(chainer.Chain):
    def __init__(self):
        super(HasItemSimple, self).__init__()
        self.x = 1

    def forward(self):
        return hasattr(self, 'x')

class HasItemProperty(chainer.Chain):
    def __init__(self):
        super(HasItemProperty, self).__init__()
        self._x = 1
    
    @property
    def x(self):
        return self._x

    def forward(self):
        return hasattr(self, 'x')


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np


def main():
    testtools.generate_testcase(HasItemSimple(), [], subname='has_item_simple')
    testtools.generate_testcase(HasItemProperty(), [], subname='has_item_property')

if __name__ == '__main__':
    main()
