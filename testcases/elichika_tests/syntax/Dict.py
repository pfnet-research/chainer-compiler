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


class DictonaryIterateKeys(chainer.Chain):
    def forward(self):
        x = {"one": 1, "two": 2, "three": 3, "four": 4}
        ret = 0
        for key in x.keys():
            ret += x[key]
        return ret


class DictonaryIterateValues(chainer.Chain):
    def forward(self):
        x = {"one": 1, "two": 2, "three": 3, "four": 4}
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

# ======================================


from chainer_compiler.elichika import testtools
import numpy as np
from itertools import product


def main():
    testtools.generate_testcase(SimpleDictionary(), [], subname='simple_dictionary')
    testtools.generate_testcase(DictionarySubscript, [], subname='dictionary_subscript')
    # testtools.generate_testcase(DictonaryIterateKeys(), [], subname='dictionary_iterate_keys')
    # testtools.generate_testcase(DictonaryIterateValues(), [], subname='dictionary_iterate_values')
    # testtools.generate_testcase(DictonaryIterateItems(), [], subname='dictionary_iterate_items')


if __name__ == '__main__':
    main()
