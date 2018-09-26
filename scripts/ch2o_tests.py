import glob
import os

from test_case import TestCase


# TODO(hamaji): Consolidate this list with the one for CMake's.

NODE_TESTS = '''
AddMul
AveragePool2d
BatchNorm
BroadcastTo
Ceil
Concat
Convolution2D
Cumsum
Dropout
EmbedID
ExpandDims
Id
LRN
Linear
MaxPool2d
NpArray
NStepLSTM
PadSequence
Relu
Reshape
Sigmoid
SoftmaxClossEntropy
SwapAxes
Tanh
'''.split()

SYNTAX_TESTS = '''
For
LinkInFor
ListComp
MultiClass
MultiFunction
Range
Sequence
Slice
UserDefinedFunc
'''.split()


def get():
    tests = []
    for category, names in [('node', NODE_TESTS),
                            ('syntax', SYNTAX_TESTS)]:
        for name in names:
            test_name = 'ch2o_%s_%s' % (category, name)
            kwargs = {}
            if ('node_BatchNorm' in test_name or
                'node_ExpandDims' in test_name or
                'node_EmbedID' in test_name or
                'node_Reshape' in test_name):
                kwargs['skip_shape_inference'] = True
            if test_name == 'ch2o_node_LRN':
                kwargs['rtol'] = 1e-3
            if test_name == 'ch2o_syntax_MultiFunction':
                kwargs['rtol'] = 1e-3

            for d in glob.glob('out/%s*' % test_name):
                tests.append(TestCase('out', os.path.basename(d), **kwargs))

    return tests
