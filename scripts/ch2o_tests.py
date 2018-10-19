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
LRN
Len
Linear
MaxPool2d
NStepBiLSTM
NStepLSTM
NpArray
NpFull
NpZeros
PadSequence
Relu
Reshape
Separate
Shape
Sigmoid
Size
SplitAxis
Softmax
SoftmaxClossEntropy
Squeeze
Stack
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
EspNet_AttDot
EspNet_BLSTM
EspNet_Decoder
MyLSTM
MLP_with_loss
StatelessLSTM
'''.split()


def get():
    tests = []

    skip_shape_inference_blacklist = [
        # ONNX's does now know cover_all.
        'node_MaxPool2d',
        # A bug of ONNX's shape inference?
        # TODO(hamaji): Investigate.
        'node_SwapAxes',
    ]

    for category, names in [('model', MODEL_TESTS),
                            ('node', NODE_TESTS),
                            ('syntax', SYNTAX_TESTS)]:
        for name in names:
            test_name = 'ch2o_%s_%s' % (category, name)
            kwargs = {}
            for substr in skip_shape_inference_blacklist:
                if substr in test_name:
                    kwargs['skip_shape_inference'] = True
                    break
            if test_name == 'ch2o_node_LRN':
                kwargs['rtol'] = 1e-3
            if test_name == 'ch2o_syntax_MultiFunction':
                kwargs['rtol'] = 1e-3

            test_dirs = glob.glob('out/%s' % test_name)
            test_dirs += glob.glob('out/%s_*' % test_name)
            for d in test_dirs:
                tests.append(TestCase('out', os.path.basename(d), **kwargs))

    return tests
