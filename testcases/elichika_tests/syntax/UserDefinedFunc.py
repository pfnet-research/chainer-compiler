# coding: utf-8

import chainer
from testcases.elichika_tests.syntax import UserDefinedFuncSub1

class F(object):
    def __init__(self, a):
        self.a = a

    def g(self, x):
        return self.a + x


def h(x, y):
    return x + y


class G(object):
    def __init__(self, a, do):
        self.a = a
        self.do = do

    def func1(self):
        temp = None
        if self.do:
            temp = self.a
        else:
            temp = self
        return temp
    
    def func2(self):
        return self

    def func3(self):
        return self.a
 
class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, y, z):
        p = F(x).g(y)
        return h(p, z)

class B(chainer.Chain):

    def __init__(self):
        super(B, self).__init__()

    def forward(self, x, y, z):
        return UserDefinedFuncSub1.h(x, y)


class C(chainer.Chain):

    def __init__(self):
        super(C, self).__init__()

    def forward(self, x, y, z):
        p = UserDefinedFuncSub1.F(x).g(y)
        return h(p, z)

class D(chainer.Chain):
    def forward(self):
        ret = 0
        obj = G(ret, False)
        ret += obj.func1().a            # TODO(rchouras): Commenting this line passes the test.
        ret += obj.func2().a
        ret += obj.func3()
        return ret


class E(chainer.Chain):
    def __init__(self):
        super(E, self).__init__()
        self.a = 1
    
    def test(self):
        self.a += 1

    def forward(self):
        self.test()
        return self.a

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np


def main():
    model = A()

    a = np.random.rand(3, 4).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(3, 4).astype(np.float32)
    testtools.generate_testcase(model, [a, b, c])
    testtools.generate_testcase(B(), [a, b, c], subname='external_func')
    testtools.generate_testcase(C(), [a, b, c], subname='external_class')
    # testtools.generate_testcase(D(), [], subname='bug')
    # testtools.generate_testcase(E(), [], subname='class_func_bug')

if __name__ == '__main__':
    main()
