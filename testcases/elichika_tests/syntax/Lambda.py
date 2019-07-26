# coding: utf-8

import chainer
import chainer.functions as F
from chainer_compiler.elichika.parser import flags

global_lambd = lambda x, y: x * y  # function name cannot contain `lambda`
global_function_product = lambda F, m: lambda x: F(x)*m

class LambdaLocalSimple(chainer.Chain):
    def forward(self, x, y):
        local_lambda_simple = lambda a, b: a * b
        return local_lambda_simple(x, y)

class LambdaLocalSimpleNested(chainer.Chain):
    def forward(self, x, y):
        square_func = lambda x: x**2
        function_product = lambda F, m: lambda x: F(x)*m
        return function_product(square_func, x)(y)

class LambdaLocalWithOutsideVariable(chainer.Chain):
    def forward(self, x, y):
        outside_variable = x - y
        local_lambda = lambda a, b: a * b + outside_variable
        return local_lambda(x, y)

class LambdaGlobalSimple(chainer.Chain):
    def forward(self, x, y):
        return global_lambd(x, y)

class LambdaGlobalNested(chainer.Chain):
    def forward(self, x, y):
        square_func = lambda x: x**2
        return global_function_product(square_func, x)(y)

class LambdaInsideIterable(chainer.Chain):
    def forward(self, x, y):
        lambdas_list = [lambda x, y: x * y, lambda x, y: x - y, lambda x, y: x + y]
        lambdas_tuple = (lambda x, y: x * y, lambda x, y: x - y, lambda x, y: x + y)
        ret = 0
        with flags.for_unroll():
            for lambda_ in lambdas_list:
                ret += lambda_(x, y)
            for lambda_ in lambdas_tuple:
                ret += lambda_(x, y)
        return ret

class LambdaConstructor(chainer.Chain):
    def __init__(self):
        super(LambdaConstructor, self).__init__()
        self.func1 = lambda a, b: a * b
        self.func2 = lambda: 3

    def forward(self, x, y):
        return self.func1(x, y) * self.func2()

class LambdaConstructorNested(chainer.Chain):
    def __init__(self):
        super(LambdaConstructorNested, self).__init__()
        self.square_func = lambda x: x**2
        self.function_product = lambda F, m: lambda x: F(x)*m

    def forward(self, x, y):
        return self.function_product(self.square_func, x)(y)



class LambdaConstructorWithOutsideVariable(chainer.Chain):
    def __init__(self):
        super(LambdaConstructorWithOutsideVariable, self).__init__()
        outside_variable = 1
        self.constructor_lambda = lambda a, b: a * b + outside_variable

    def forward(self, x, y):
        return self.constructor_lambda(x, y)


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np

def main():
    x, y = 3, 4

    testtools.generate_testcase(LambdaLocalSimple, [x, y], subname='lambda_local_simple')
    # testtools.generate_testcase(LambdaLocalSimpleNested, [x, y], subname='lambda_local_nested')
    testtools.generate_testcase(LambdaLocalWithOutsideVariable, [x, y], subname='lambda_local_with_outside_variable')
    testtools.generate_testcase(LambdaGlobalSimple, [x, y], subname='lambda_global_simple')
    # testtools.generate_testcase(LambdaGlobalNested, [x, y], subname='lambda_global_nested')
    # testtools.generate_testcase(LambdaInsideIterable, [x, y], subname='lambda_inside_iterable')
    testtools.generate_testcase(LambdaConstructor, [x, y], subname='lambda_constructor')
    # testtools.generate_testcase(LambdaConstructorNested, [x, y], subname='lambda_constructor_nested')
    # testtools.generate_testcase(LambdaConstructorWithOutsideVariable, [x, y], subname='lambda_constructor_with_outside_variable')


if __name__ == '__main__':
    main()
