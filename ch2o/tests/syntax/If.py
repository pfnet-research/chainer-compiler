# coding: utf-8

import chainer
import chainer.functions as F


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


# ======================================


import ch2o
import numpy as np


if __name__ == '__main__':
    ch2o.generate_testcase(StaticCondTrue(), [42])

    ch2o.generate_testcase(StaticCondFalse(), [42], subname='static_false')

    ch2o.generate_testcase(StaticCondTrueNoElse(), [42],
                           subname='static_true_no_else')

    ch2o.generate_testcase(StaticCondFalseNoElse(), [42],
                           subname='static_false_no_else')

    ch2o.generate_testcase(DynamicCond(), [42, False], subname='false')

    ch2o.generate_testcase(DynamicCond(), [42, True], subname='true')

    ch2o.generate_testcase(DynamicCondNoElse(), [42, False],
                           subname='false_no_else')

    ch2o.generate_testcase(DynamicCondNoElse(), [42, True],
                           subname='true_no_else')

    ch2o.generate_testcase(DynamicCondLeak(), [42, False], subname='leak_false')

    ch2o.generate_testcase(DynamicCondLeak(), [42, True], subname='leak_true')

    ch2o.generate_testcase(DynamicCondAlias(), [42, False],
                           subname='alias_false')

    ch2o.generate_testcase(DynamicCondAlias(), [42, True],
                           subname='alias_true')

    ch2o.generate_testcase(UpdateSelf(), [42, True],
                           subname='update_self_true')
    ch2o.generate_testcase(UpdateSelf(), [42, False],
                           subname='update_self_false')

    ch2o.generate_testcase(LazyInit, [10],
                           subname='lazy_init')

    ch2o.generate_testcase(LazyInitUse, [10],
                           subname='lazy_init_use')
