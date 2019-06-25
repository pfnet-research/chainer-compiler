# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class StaticCondTrue(chainer.Chain):
    def forward(self, x):
        if True:
            x += 3
        else:
            x += 10
        return x


class StaticCondFalse(chainer.Chain):
    def forward(self, x):
        if False:
            x += 3
        else:
            x += 10
        return x


class StaticCondTrueNoElse(chainer.Chain):
    def forward(self, x):
        if True:
            x += 3
        return x


class StaticCondFalseNoElse(chainer.Chain):
    def forward(self, x):
        if False:
            x += 3
        return x


class DynamicCond(chainer.Chain):
    def forward(self, x, cond):
        if cond:
            x += 3
        else:
            x += 10
        return x


class DynamicCondNoElse(chainer.Chain):
    def forward(self, x, cond):
        if cond:
            x += 3
        return x


class DynamicCondLeak(chainer.Chain):
    def forward(self, x, cond):
        if cond:
            y = x + 3
        else:
            y = x + 10
        return y


class DynamicCondAlias(chainer.Chain):
    def forward(self, x, cond):
        y = x
        if cond:
            y += 3
        else:
            y += 10
        return y


class UpdateSelf(chainer.Chain):
    def forward(self, x, cond):
        self.x = x
        if cond:
            self.x += 10
        return self.x


class LazyInit(chainer.Chain):
    def __init__(self):
        super(LazyInit, self).__init__()
        self.x = None

    def forward(self, x):
        if self.x is None:
            self.x = x
        return self.x


class LazyInitUse(chainer.Chain):
    def __init__(self):
        super(LazyInitUse, self).__init__()
        with self.init_scope():
            self.li = LazyInit()

    def forward(self, x):
        a = self.li(x)
        b = self.li(x * 2)
        return a + b


class IfBackprop(chainer.Chain):
    def __init__(self):
        super(IfBackprop, self).__init__()
        with self.init_scope():
            self.l = L.Linear(None, 4)

    def forward(self, x, cond):
        if cond is not None:
            x = self.l(x)
        return x

c = 2
class Global(chainer.Chain):
    def __init__(self):
        super(Global, self).__init__()

    def forward(self, c, x):
        y = x
        if c is None:
            c = 1
        return (y,c)

def func(y):
    return y + 1

class UserFunc(chainer.Chain):
    def __init__(self):
        super(UserFunc, self).__init__()
        self.w = None

    def forward(self, z):
        if z is not None:
            self.w = func(z)
        return self.w

class NpArrayShape(chainer.Chain):
    def __init__(self):
        super(NpArrayShape, self).__init__()

    def forward(self, x):
        if x is not None:
            x = F.relu(x)
        return list(x.shape)

class NpArrayShapeSelf(chainer.Chain):
    def __init__(self):
        super(NpArrayShapeSelf, self).__init__()

    def forward(self, x):
        if x is not None:
            self.x = F.relu(x)
        return list(self.x.shape)

def func_array(arg0):
    y = F.relu(arg0)
    return y

class LazyInitWithFunc(chainer.Chain):
    def __init__(self):
        super(LazyInitWithFunc, self).__init__()
        self.x = None

    def forward(self, x):
        if self.x is None:
            self.x = func_array(x)
        return self.x

class LazyInitWithFuncUse(chainer.Chain):
    def __init__(self):
        super(LazyInitWithFuncUse, self).__init__()
        with self.init_scope():
            self.li = LazyInitWithFunc()

    def forward(self, x):
        a = self.li(x)
        b = self.li(a * 2)
        return a + b

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np


def main():
    testtools.generate_testcase(StaticCondTrue(), [42], subname='static_true')

    testtools.generate_testcase(StaticCondFalse(), [42], subname='static_false')

    testtools.generate_testcase(StaticCondTrueNoElse(), [42],
                           subname='static_true_no_else')

    testtools.generate_testcase(StaticCondFalseNoElse(), [42],
                           subname='static_false_no_else')
    
    testtools.generate_testcase(DynamicCond(), [42, False], subname='false')

    testtools.generate_testcase(DynamicCond(), [42, True], subname='true')

    testtools.generate_testcase(DynamicCondNoElse(), [42, False],
                           subname='false_no_else')

    testtools.generate_testcase(DynamicCondNoElse(), [42, True],
                           subname='true_no_else')
    
    testtools.generate_testcase(DynamicCondLeak(), [42, False], subname='leak_false')

    testtools.generate_testcase(DynamicCondLeak(), [42, True], subname='leak_true')

    testtools.generate_testcase(DynamicCondAlias(), [42, False],
                           subname='alias_false')

    testtools.generate_testcase(DynamicCondAlias(), [42, True],
                           subname='alias_true')

    testtools.generate_testcase(UpdateSelf(), [42, True],
                           subname='update_self_true')
    testtools.generate_testcase(UpdateSelf(), [42, False],
                           subname='update_self_false')

    testtools.generate_testcase(LazyInit, [10],
                           subname='lazy_init')

    testtools.generate_testcase(LazyInitUse, [10],
                           subname='lazy_init_use')


    model_IfBackprop = IfBackprop()
    model_IfBackprop(np.random.rand(3, 5).astype(np.float32), 1)
    testtools.generate_testcase(model_IfBackprop,
                           [np.random.rand(3, 5).astype(np.float32), 1],
                           subname='if_bp', backprop=True)
    
    n = np.random.rand(2, 2).astype(np.float32)
    c = np.random.rand(2, 2).astype(np.float32)
    testtools.generate_testcase(Global(), [c,n], subname='global')

    testtools.generate_testcase(UserFunc(), [42], subname='user_func')

    testtools.generate_testcase(NpArrayShape, [n], subname='nparray_shape')

    testtools.generate_testcase(NpArrayShapeSelf, [n], subname='nparray_shape_self')

    testtools.generate_testcase(LazyInitWithFuncUse, [n], subname='lazy_init_with_func_use')

if __name__ == '__main__':
    main()
