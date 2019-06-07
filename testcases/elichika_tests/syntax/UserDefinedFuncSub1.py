# coding: utf-8

import chainer
from testcases.elichika_tests.syntax import UserDefinedFuncSub2

def h(x, y):
    return x + UserDefinedFuncSub2.h(x, x + y)

def i(x, y):
    return x + UserDefinedFuncSub2.h(x, x + y)

class F(object):
    def __init__(self, a):
        self.a = a

    def g(self, x):
        return i(self.a, x)


