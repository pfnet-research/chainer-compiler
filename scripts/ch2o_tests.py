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
Hstack
Id
Len
LRN
Linear
MaxPool2d
NpArray
NpZeros
NStepBiLSTM
NStepLSTM
PadSequence
Relu
Reshape
Sigmoid
SplitAxis
Softmax
SoftmaxClossEntropy
Sum
SwapAxes
Tanh
Variable
Vstack
'''.split()

SYNTAX_TESTS = '''
Cmp
For
ForAndIf
If
LinkInFor
ListComp
MultiClass
MultiFunction
Range
Sequence
Slice
UserDefinedFunc
'''.split()

MODEL_TESTS = '''
EspNet_BLSTM
EspNet_AttDot
MyLSTM
MLP_with_loss
'''.split()


def get():
    tests = []
    for category, names in [('model', MODEL_TESTS),
                            ('node', NODE_TESTS),
                            ('syntax', SYNTAX_TESTS)]:
        for name in names:
            test_name = 'ch2o_%s_%s' % (category, name)
            kwargs = {}
            # TODO(hamaji): Get rid of this whitelist or reduce the
            # number of tests at least by setting input/output types
            # properly.
            if ('node_BatchNorm' in test_name or
                'node_ExpandDims' in test_name or
                'node_EmbedID' in test_name or
                'node_Size' in test_name or
                'node_Sum' in test_name or
                'node_Reshape' in test_name):
                kwargs['skip_shape_inference'] = True
            if test_name == 'ch2o_node_LRN':
                kwargs['rtol'] = 1e-3
            if test_name == 'ch2o_syntax_MultiFunction':
                kwargs['rtol'] = 1e-3

            test_dirs = glob.glob('out/%s' % test_name)
            test_dirs += glob.glob('out/%s_*' % test_name)
            for d in test_dirs:
                tests.append(TestCase('out', os.path.basename(d), **kwargs))

    return tests
