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
        test_list[2] += 2
        return test_list


class IfListSubscriptAssign(chainer.Chain):
    def forward(self):
        test_list = [0, 1, 2]
        if True:
            test_list = test_list[1]
        return test_list

class ArraySubscript(chainer.Chain):
    def forward(self):
        test_list = np.array([0, 1, 2])
        x = test_list[2]
        test_list[1] = x
        #TODO(rchouras): Debug error
        # x = test_list[0:1]
        # test_list[1:2] = x
        return test_list

class ArraySubscriptFancy(chainer.Chain):
    def forward(self, x, y):
        # (2, 3, 4, 5) => (2, 3, 4)
        y[1,2,3,4]=x[0,1,2,3]
        t = x[:, 1, :3, -4:]
        return t, y
        # TODO(hamaji): Support multi-axes subsription.
        # y[1, 1:3, :3, :4] = t
        # return t, y
        # TODO(hamaji): Support fancy indexing by ndarray.
        # y = test_list[:, 1, np.array([2, 1]), :-4:]
        # return x, y

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
        x = [0]
        x.append(x)
        return x[1][1][1][0]

class TensorToList(chainer.Chain):
    def forward(self, tensor):
        x = list(tensor)
        x.append(1)
        return x

# ======================================


from chainer_compiler.elichika import testtools
import numpy as np
from itertools import product


def main():
    testtools.generate_testcase(SimpleList(), [], subname='simple_list')
    testtools.generate_testcase(ListIterate(), [], subname='list_iterate')
    testtools.generate_testcase(ListInConstructor(), [], subname='list_in_constructor')

    # TODO(rchouras): Fix following tests. First two are used very commonly.
    testtools.generate_testcase(ListSubscript, [], subname='list_subscript')
    testtools.generate_testcase(IfListSubscriptAssign, [], subname='if_list_subscript_assign')
    testtools.generate_testcase(ArraySubscript, [], subname='array_subscript')
    x, y = (np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)) for _ in range(2))
    testtools.generate_testcase(ArraySubscriptFancy, [x, y],
                                subname='array_subscript_fancy')
    # testtools.generate_testcase(ListAssignByValueRef, [], subname='list_assign_by_value_ref')
    # testtools.generate_testcase(ListInfinitelyNested(), [], subname='list_nested')

    tensor = np.array([1, 2, 3, 4, 5])
    testtools.generate_testcase(TensorToList(), [tensor], subname='tensor_to_list')

if __name__ == '__main__':
    main()
