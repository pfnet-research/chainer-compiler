# coding: utf-8

from chainer_compiler.elichika import testtools
import numpy as np
import chainer


class RetTuple(chainer.Chain):

    def __init__(self):
        super(RetTuple, self).__init__()

    def forward(self):
        return (1, 2)


class SubstituteTuple(chainer.Chain):
    def __init__(self):
        super(SubstituteTuple, self).__init__()

    def get_tuple(self):
        return (1, 2)

    def forward(self):
        x, y = self.get_tuple()
        return x


class SubstituteTupleSelf(chainer.Chain):
    def __init__(self):
        super(SubstituteTupleSelf, self).__init__()
        self.a = 1

    def get_tuple(self):
        return (self.a + 1, self.a + 2)

    def forward(self):
        x, y = self.get_tuple()
        return x

# ======================================


def main():
    testtools.generate_testcase(RetTuple(), [], subname='ret_tuple')
    testtools.generate_testcase(
        SubstituteTuple(), [], subname='substitute_tuple')

    testtools.generate_testcase(
        SubstituteTupleSelf(), [], subname='substitute_tuple_self')


if __name__ == '__main__':
    main()
