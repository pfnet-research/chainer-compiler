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
Matmul
MaxPool2d
Mean
NStepBiLSTM
NStepLSTM
NpArray
NpFull
NpZeros
PadSequence
Relu
Reshape
ResizeImages
Roi
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
Unpooling2D
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
EspNet_AttLoc
EspNet_BLSTM
EspNet_Decoder
EspNet_E2E
EspNet_VGG2L
MyLSTM
MLP_with_loss
StatelessLSTM
'''.split()


def get():
    tests = []

    skip_shape_inference_blacklist = [
        # A bug of ONNX's shape inference?
        # TODO(hamaji): Investigate.
        'node_SwapAxes',
    ]

    diversed_whitelist = [
        'node_Linear'
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

            diversed = False
            for substr in diversed_whitelist:
                if substr in test_name:
                    diversed = True
                    break

            test_dirs = glob.glob('out/%s' % test_name)
            test_dirs += glob.glob('out/%s_*' % test_name)
            for d in test_dirs:
                name = os.path.basename(d)
                test_dir = os.path.join('out', name)
                tests.append(TestCase(name=name, test_dir=test_dir, **kwargs))
                if diversed:
                    tests.append(TestCase(name=name + '_diversed',
                                          test_dir=test_dir,
                                          backend='chxvm_test',
                                          **kwargs))

    return tests
