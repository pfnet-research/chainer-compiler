# coding: utf-8

import chainer


class SimpleList(chainer.Chain):
    def forward(self):
        x = [1, 2]
        y = [0, 1, 2]
        return x[0] * y[1]


class ListSubscript(chainer.Chain):
    def forward(self):
        test_list = [0, 1, 2] 
        test_list[1] = 3
        return test_list

class ListAssignByValueRef(chainer.Chain):
    def forward(self):
        # shared reference
        list_ = [1]
        test_list1 = [list_]
        test_list1[0].append(2)
        # shared value
        num_ = 1
        test_list2 = [num_]
        test_list2[0] = 2
        return test_list1[0][1] * test_list2[0]

class ListIterate(chainer.Chain):
    def forward(self):
        x = [1, 2, 3, 4, 5]
        x.append(6)
        ret = 0
        for value in range(len(x)):
            ret += x[value]
        return ret

class ListInConstructor(chainer.Chain):
    def __init__(self):
        super(ListInConstructor, self).__init__()
        self.test_list = [1, 2]

    def forward(self):
        return self.test_list[0]

class ListInfinitelyNested(chainer.Chain):
    def forward(self):
        x = [1]
        x.append(x)
        return x[1][1][1][0]


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np
from itertools import product


def main():
    testtools.generate_testcase(SimpleList(), [], subname='simple_list')
    testtools.generate_testcase(ListIterate(), [], subname='list_iterate')
    testtools.generate_testcase(ListInConstructor(), [], subname='list_in_constructor')

    # TODO(rchouras): Fix following tests. First two are used very commonly.
    # testtools.generate_testcase(ListSubscript, [], subname='list_subscript')
    # testtools.generate_testcase(ListAssignByValueRef, [], subname='list_assign_by_value_ref')
    # testtools.generate_testcase(ListInfinitelyNested(), [], subname='list_nested')


if __name__ == '__main__':
    main()
