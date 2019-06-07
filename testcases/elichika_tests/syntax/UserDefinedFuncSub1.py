# coding: utf-8

import chainer
import tests.syntax.UserDefinedFuncSub2

def h(x, y):
    return x + tests.syntax.UserDefinedFuncSub2.h(x, x + y)

def i(x, y):
    return x + tests.syntax.UserDefinedFuncSub2.h(x, x + y)

class F(object):
    def __init__(self, a):
        self.a = a

    def g(self, x):
        return i(self.a, x)


