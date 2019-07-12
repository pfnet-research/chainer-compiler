# coding: utf-8

import chainer


class SimpleDictionary(chainer.Chain):
    def forward(self):
        x = {"one": 1, "two": 2}
        y = {1: 1, 2: 2, 1.0: 3}
        return x["one"] * y[1]


class DictionarySubscript(chainer.Chain):
    def forward(self):
        test_dict = {}
        value = 1
        for key in [True, 2, "Three", 2, True, 1]:
            test_dict[key] = value
            value += 1
        return test_dict[True] * test_dict[2] * test_dict["Three"]

class DictionaryAssignByValueRef(chainer.Chain):
    def forward(self):
        # shared reference
        list_ = [1]
        test_dict1 = {"x": list_, "y": list_}
        test_dict1["x"].append(2)
        # shared value
        num_ = 1
        test_dict2 = {"x": num_, "y": num_}
        test_dict2["x"] += 1
        return test_dict1["y"][1] * test_dict2["y"]

class DictonaryIterateKeys(chainer.Chain):
    def forward(self):
        x = {1: 1, "two": 2, 3: 3, "four": 4, True: 5}
        x["five"] = 5
        ret = 0
        for key in x.keys():
            ret += x[key]
        return ret


class DictonaryIterateValues(chainer.Chain):
    def forward(self):
        x = {1: 1, "two": 2, 3: 3, "four": 4, True: 5}
        x["five"] = 5
        ret = 0
        for value in x.values():
            ret += value
        return ret


class DictonaryIterateItems(chainer.Chain):
    def forward(self):
        x = {"one": 1, "two": 2, "three": 3, "four": 4}
        ret = 0
        for key, value in x.items():
            ret += value
        return ret

class DictionaryInConstructor(chainer.Chain):
    def __init__(self):
        super(DictionaryInConstructor, self).__init__()
        self.test_dict = {"x": 1, "y" : 2}

    def forward(self):
        return self.test_dict["x"]

class DictionaryInfinitelyNested(chainer.Chain):
    def forward(self):
        x = {"one": 1}
        x["two"] = x
        return x["two"]["two"]["two"]["one"]


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np
from itertools import product


def main():
    testtools.generate_testcase(SimpleDictionary(), [], subname='simple_dictionary')
    testtools.generate_testcase(DictionarySubscript, [], subname='dictionary_subscript')
    testtools.generate_testcase(DictionaryAssignByValueRef, [], subname='assign_by_value_ref')
    testtools.generate_testcase(DictonaryIterateKeys(), [], subname='dictionary_iterate_keys')
    testtools.generate_testcase(DictonaryIterateValues(), [], subname='dictionary_iterate_values')
    testtools.generate_testcase(DictionaryInfinitelyNested(), [], subname='dictionary_nested')
    testtools.generate_testcase(DictionaryInConstructor(), [], subname='dictionary_in_constructor')
    # testtools.generate_testcase(DictonaryIterateItems(), [], subname='dictionary_iterate_items')


if __name__ == '__main__':
    main()
