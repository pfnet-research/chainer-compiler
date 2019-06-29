# coding: utf-8

import chainer


class SimpleDictionary(chainer.Chain):
    def forward(self):
        x = {"one": 1, "two": 2}
        return x["one"] * x["two"]


class DictionarySubscript(chainer.Chain):
    def forward(self, key, value):
        test_dict = {key: None}
        test_dict[key] = value
        return test_dict[key]


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
    testtools.generate_testcase(DictonaryIterateKeys(), [], subname='dictionary_iterate_keys')
    testtools.generate_testcase(DictonaryIterateValues(), [], subname='dictionary_iterate_values')
    testtools.generate_testcase(DictonaryIterateItems(), [], subname='dictionary_iterate_items')

    for key, value in product([1, "one", True], repeat=2):
        testtools.generate_testcase(DictionarySubscript, [key, value], subname='dictionary_subscript_%s_%s' % (str(key), str(value)))



if __name__ == '__main__':
    main()
