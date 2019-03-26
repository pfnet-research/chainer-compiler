import collections


ARRAY = 'ARRAY'
OPTIONAL_ARRAY = 'OPTIONAL_ARRAY'
ARRAY_LIST = 'ARRAY_LIST'
SEQUENCE = 'SEQUENCE'
OPAQUE = 'OPAQUE'
INT = 'INT'
FLOAT = 'FLOAT'
INTS = 'INTS'
STRING = 'STRING'
DOUBLES = 'DOUBLES'

ARG_TYPES = [
    ARRAY, OPTIONAL_ARRAY, ARRAY_LIST, SEQUENCE, OPAQUE
]

FIELD_TYPES = [
    INT, FLOAT, INTS, STRING, DOUBLES
]

XC_TYPES = ARG_TYPES + FIELD_TYPES

STACK_VECTOR = 'chainerx::StackVector<int64_t, chainerx::kMaxNdim>'


_ValueInfo = collections.namedtuple('_ValueInfo', ('typ', 'name'))


class ValueInfo(_ValueInfo):
    def is_repeated(self):
        return self.typ in [INTS, ARRAY_LIST, DOUBLES]

    def c_type(self):
        if self.typ in [ARRAY, OPTIONAL_ARRAY, INT, SEQUENCE, OPAQUE]:
            return 'int'
        elif self.typ == FLOAT:
            return 'float'
        elif self.typ == STRING:
            return 'std::string'
        elif self.typ == ARRAY_LIST:
            return 'std::vector<int>'
        elif self.typ == INTS:
            return 'std::vector<int64_t>'
        elif self.typ == DOUBLES:
            return 'std::vector<double>'

    def c_storage_type(self):
        if self.typ == INTS:
            return STACK_VECTOR
        else:
            return self.c_type()

    def c_arg_type(self):
        ctyp = self.c_type()
        if 'std::' in ctyp:
            return 'const %s&' % (ctyp)
        return ctyp

    def c_codegen_type(self):
        if self.typ in (ARRAY, OPTIONAL_ARRAY, SEQUENCE, OPAQUE):
            return 'XCVMValue'
        elif self.typ == ARRAY_LIST:
            return 'std::vector<XCVMValue>'
        else:
            return self.c_type()

    def proto_field_name(self):
        if self.typ in [ARRAY, OPTIONAL_ARRAY]:
            return 'array'
        elif self.typ == SEQUENCE:
            return 'sequence'
        elif self.typ == INT:
            return 'i'
        elif self.typ == FLOAT:
            return 'f'
        elif self.typ == STRING:
            return 's'
        elif self.typ == INTS:
            return 'ints'
        elif self.typ == DOUBLES:
            return 'doubles'
        elif self.typ == ARRAY_LIST:
            return 'array_list'
        elif self.typ == OPAQUE:
            return 'opaque'
        else:
            raise RuntimeError('Unknown type: %s' % self.typ)


def Array(name):
    return ValueInfo(ARRAY, name)


def OptionalArray(name):
    return ValueInfo(OPTIONAL_ARRAY, name)


def ArrayList(name):
    return ValueInfo(ARRAY_LIST, name)


def Sequence(name):
    return ValueInfo(SEQUENCE, name)


def Opaque(name):
    return ValueInfo(OPAQUE, name)


def Int(name):
    return ValueInfo(INT, name)


def Float(name):
    return ValueInfo(FLOAT, name)


def Ints(name):
    return ValueInfo(INTS, name)


def String(name):
    return ValueInfo(STRING, name)


def Doubles(name):
    return ValueInfo(DOUBLES, name)


def sigil(typ):
    if typ in [ARRAY, OPTIONAL_ARRAY]:
        return '$'
    elif typ == SEQUENCE:
        return '@'
    elif typ == OPAQUE:
        return '*'
    else:
        raise RuntimeError('Not a varaible: %s' % typ)


XC_OPS = [
    ('Add', [Array('a'), Array('b')], ['c']),
    ('Sub', [Array('a'), Array('b')], ['c']),
    ('Mul', [Array('a'), Array('b')], ['c']),
    ('Div', [Array('a'), Array('b')], ['c']),
    ('Pow', [Array('a'), Array('b')], ['c']),
    ('Neg', [Array('x')], ['y']),
    ('And', [Array('a'), Array('b')], ['c']),
    ('Or', [Array('a'), Array('b')], ['c']),
    ('Xor', [Array('a'), Array('b')], ['c']),

    ('Reciprocal', [Array('x')], ['y']),
    ('Exp', [Array('x')], ['y']),
    ('Log', [Array('x')], ['y']),
    ('Sqrt', [Array('x')], ['y']),
    ('Abs', [Array('x')], ['y']),
    ('Sin', [Array('x')], ['y']),
    ('Sinh', [Array('x')], ['y']),
    ('Cos', [Array('x')], ['y']),
    ('Cosh', [Array('x')], ['y']),
    ('Tan', [Array('x')], ['y']),
    ('Tanh', [Array('x')], ['y']),
    ('Asin', [Array('x')], ['y']),
    ('Asinh', [Array('x')], ['y']),
    ('Acos', [Array('x')], ['y']),
    ('Acosh', [Array('x')], ['y']),
    ('Atan', [Array('x')], ['y']),
    ('Atanh', [Array('x')], ['y']),
    ('Sigmoid', [Array('x')], ['y']),

    ('ArgMax', [Array('x'), Int('axis'), Int('keepdims')], ['y']),
    ('Hardmax', [Array('x'), Int('axis')], ['y']),

    ('Clip', [Array('inputs'), Float('max'), Float('min')], ['result']),
    ('Max', [ArrayList('inputs')], ['result']),
    ('ReduceMax', [Array('x'), Ints('axes'), Int('keepdims')], ['y']),
    ('ReduceSum', [Array('data'), Ints('axes'), Int('keepdims')], ['reduced']),
    ('ReduceSumSquare', [Array('data'), Ints('axes'), Int('keepdims')],
     ['reduced']),
    ('ReduceSumTo', [Array('data'), Array('shape')], ['reduced']),
    ('ReduceMean', [Array('data'), Ints('axes'), Int('keepdims')], ['reduced']),

    ('Linear',
     [Array('x'), Array('w'), OptionalArray('b'), Int('n_batch_axes')],
     ['y']),
    ('LinearGradWeight', [Array('x'), Array('gy')], ['gw']),

    ('Conv',
     [Array('x'), Array('w'), OptionalArray('b'),
      Ints('strides'), Ints('pads')], ['y']),
    ('ConvTranspose',
     [Array('x'), Array('w'), OptionalArray('b'),
      Ints('strides'), Ints('pads'), Ints('output_shape')], ['y']),
    ('ConvTransposeWithDynamicShape',
     [Array('x'), Array('w'), Array('output_shape'),
      Ints('strides'), Ints('pads')], ['y']),
    ('ConvGradWeight',
     [Array('w'), Array('x'), Array('gy'), Ints('strides'), Ints('pads')],
     ['y']),

    ('Relu', [Array('x')], ['y']),
    ('ReluGrad', [Array('x'), Array('gy')], ['gx']),
    ('Selu', [Array('x'), Float('alpha'), Float('gamma')], ['y']),
    ('LeakyRelu', [Array('x'), Float('alpha')], ['y']),
    ('Elu', [Array('x'), Float('alpha')], ['y']),
    ('Floor', [Array('x')], ['y']),
    ('Ceil', [Array('x')], ['y']),
    ('Shape', [Array('data')], ['shape']),
    ('Size', [Array('data')], ['size']),
    ('Reshape', [Array('data'), Array('shape')], ['reshaped']),
    ('Expand', [Array('input'), Array('shape')], ['output']),
    ('Squeeze', [Array('data'), Ints('axes')], ['squeezed']),
    ('Unsqueeze', [Array('data'), Ints('axes')], ['expanded']),
    ('Slice', [Array('data'), Ints('axes'), Ints('starts'), Ints('ends')],
     ['output']),
    ('DynamicSlice',
     [Array('data'), Array('starts'), Array('ends'),
      OptionalArray('axes'), OptionalArray('steps')],
     ['output']),
    ('DynamicSliceGrad',
     [Array('gy'), Array('shape'), Array('starts'), Array('ends'),
      OptionalArray('axes'), OptionalArray('steps')],
     ['gx']),
    ('GetItem', [Array('data'), ArrayList('slices'), Ints('slice_specs')],
     ['output']),
    ('GetItemGrad',
     [Array('gy'), Array('shape'), ArrayList('slices'), Ints('slice_specs')],
     ['output']),
    ('Gather', [Array('data'), Array('indices'), Int('axis')], ['output']),
    ('GatherGrad',
     [Array('gy'), Array('indices'), Array('shape'), Int('axis')], ['gx']),
    ('SelectItem', [Array('data'), Array('indices')], ['output']),
    ('SelectItemGrad', [Array('gy'), Array('indices'), Array('shape')], ['gx']),
    ('Concat', [ArrayList('inputs'), Int('axis')], ['concat_result']),
    ('Split', [Array('input'), Int('axis'), Ints('split')],
     [ArrayList('outputs')]),
    ('Transpose', [Array('data'), Ints('perm')], ['transposed']),
    ('DepthToSpace', [Array('input'), Int('blocksize')], ['output']),
    ('SpaceToDepth', [Array('input'), Int('blocksize')], ['output']),

    ('Softmax', [Array('input'), Int('axis')], ['output']),
    ('LogSoftmax', [Array('input'), Int('axis')], ['output']),

    ('Dropout', [Array('data'), Float('ratio')], ['output', 'mask']),

    ('Upsample', [Array('x'), Array('scales')], ['y']),
    ('Pad', [Array('data'), Ints('pads'), Float('value')], ['output']),
    ('MaxPool',
     [Array('x'), Ints('kernel_shape'), Ints('strides'), Ints('pads'),
      Int('cover_all')],
     ['y', Opaque('ctx')]),
    ('AveragePool',
     [Array('x'), Ints('kernel_shape'), Ints('strides'), Ints('pads'),
      Int('count_include_pad')],
     ['y', Opaque('ctx')]),
    ('MaxPoolGrad', [Array('gy'), Opaque('ctx')], ['gx']),
    ('AveragePoolGrad', [Array('gy'), Opaque('ctx')], ['gx']),
    ('MaxPoolGradNoCtx',
     [Array('x'), Array('y'), Array('gy'),
      Ints('kernel_shape'), Ints('strides'), Ints('pads'), Int('cover_all')],
     ['gx']),
    ('AveragePoolGradNoCtx',
     [Array('x'), Array('y'), Array('gy'),
      Ints('kernel_shape'), Ints('strides'), Ints('pads'),
      Int('count_include_pad')],
     ['gx']),

    ('ROIMaxPool2D',
     [Array('x'), Array('rois'), Array('roi_indices'),
      Ints('output_shape'), Float('spatial_scale')],
     ['y']),
    ('ROIAveragePool2D',
     [Array('x'), Array('rois'), Array('roi_indices'),
      Ints('output_shape'), Float('spatial_scale')],
     ['y']),
    ('ROIMaxAlign2D',
     [Array('x'), Array('rois'), Array('roi_indices'),
      Ints('output_shape'), Float('spatial_scale'), Ints('sampling_ratio')],
     ['y']),
    ('ROIAverageAlign2D',
     [Array('x'), Array('rois'), Array('roi_indices'),
      Ints('output_shape'), Float('spatial_scale'), Ints('sampling_ratio')],
     ['y']),
    ('ResizeImages',
     [Array('x'), Ints('output_shape')],
     ['y']),

    ('MatMul', [Array('a'), Array('b')], ['y']),
    ('Gemm',
     [Array('a'), Array('b'), Array('c'),
      Float('alpha'), Float('beta'), Int('trans_a'), Int('trans_b')],
     ['y']),

    ('RNN',
     [Array('x'), Array('w'), Array('r'),
      OptionalArray('b'), OptionalArray('sequence_lens'),
      OptionalArray('initial_h'),
      Int('hidden_size'), Int('direction'),
     ],
     ['y', 'y_h']),
    ('GRU',
     [Array('x'), Array('w'), Array('r'),
      OptionalArray('b'), OptionalArray('sequence_lens'),
      OptionalArray('initial_h'),
      Int('hidden_size'), Int('linear_before_reset'), Int('direction'),
     ],
     ['y', 'y_h']),
    ('LSTM',
     [Array('x'), Array('w'), Array('r'),
      OptionalArray('b'), OptionalArray('sequence_lens'),
      OptionalArray('initial_h'), OptionalArray('initial_c'),
      OptionalArray('p'),
      Int('hidden_size'), Int('direction'),
     ],
     ['y', 'y_h', 'y_c', Opaque('ctx')]),
    ('LSTMGrad',
     [Array('gy'), Opaque('ctx')],
     ['gx', 'gw', 'gr', 'gb']),

    ('BatchNormalization',
     [Array('x'), Array('s'), Array('bias'), Array('mean'), Array('var'),
      Float('epsilon'), Float('decay')],
     ['y', Opaque('ctx'),
      OptionalArray('running_mean'), OptionalArray('running_var'),
      OptionalArray('saved_mean'), OptionalArray('saved_var')]),
    ('FixedBatchNormalization',
     [Array('x'), Array('s'), Array('bias'), Array('mean'), Array('var'),
      Float('epsilon')],
     ['y']),
    ('BatchNormalizationGrad', [Array('gy'), Opaque('ctx')],
     ['gx0', 'gx1', 'gx2']),

    ('LRN',
     [Array('x'), Float('alpha'), Float('beta'), Float('bias'), Int('size')],
     ['y', 'unit_scale']),
    ('LRNGrad',
     [Array('x'), Array('y'), Array('gy'), Array('unit_scale'),
      Float('alpha'), Float('beta'), Float('bias'), Int('size')], ['gx']),

    ('Equal', [Array('a'), Array('b')], ['c']),
    ('Greater', [Array('a'), Array('b')], ['c']),
    ('GreaterEqual', [Array('a'), Array('b')], ['c']),
    ('Not', [Array('x')], ['y']),
    ('Cast', [Array('input'), Int('to')], ['output']),

    ('IntScalarConstant',
     [Int('value'), Int('dtype'), Int('host')], ['output']),
    ('FloatScalarConstant',
     [Float('value'), Int('dtype'), Int('host')], ['output']),
    ('IntConstant',
     [Ints('value'), Int('dtype'), Ints('shape'), Int('host')], ['output']),
    ('FloatConstant',
     [Doubles('value'), Int('dtype'), Ints('shape'), Int('host')], ['output']),
    ('ConstantFill',
     [OptionalArray('input'), Int('dtype'), Ints('extra_shape'),
      Ints('shape'), Float('value')],
     ['output']),
    ('OneHot',
     [Array('indices'), Array('depth'), Array('values'), Int('axis')],
     ['output']),
    ('EyeLike', [Array('input'), Int('dtype'), Int('k')], ['output']),

    ('Jmp', [Int('pc')], []),
    ('JmpTrue', [Array('cond'), Int('pc')], []),
    ('JmpFalse', [Array('cond'), Int('pc')], []),

    ('ElementWiseNvrtc',
     [ArrayList('inputs'), Int('num_outputs'),
      String('code'), Int('fusion_id')],
     [ArrayList('outputs')]),

    ('DoSomething',
     [ArrayList('inputs'), String('func_name')],
     [ArrayList('outputs')]),
]

XC_CUSTOM_FIELD_OPS = [
    ('TVM',
     [ArrayList('inputs'), Int('num_outputs'),
      String('dso_filename'), String('func_name'), Ints('output_shape')],
     [ArrayList('outputs')]),
    ('NGraph',
     [ArrayList('inputs'), String('onnx')],
     [ArrayList('outputs')]),
]

XC_SEQ_OPS = [
    ('SequenceCreate', [ArrayList('inputs')], [Sequence('output')]),
    ('SequenceLookup', [Sequence('seq'), Array('index')], [Array('output')]),
    ('SequenceLookupGrad', [Array('gy'), Array('size'), Array('index')],
     [Sequence('gx')]),
    ('SequenceGetSlice',
     [Sequence('seq'), OptionalArray('start'),
      OptionalArray('end'), OptionalArray('step')],
     [Sequence('output')]),
    ('SequenceGetSliceGrad',
     [Sequence('gy'), Array('size'),
      OptionalArray('start'), OptionalArray('end'), OptionalArray('step')],
     [Sequence('gx')]),
    ('SequenceStack', [Sequence('seq'), Int('axis')], ['output']),
    ('SequenceConcat', [Sequence('seq'), Int('axis')],
     ['output', 'ctx']),
    ('SequenceSplitAxis',
     [Array('seq'), Array('indices_or_sections'), Int('axis')],
     [Sequence('output')]),
    ('SequencePad', [Sequence('seq'), Int('length'), Float('value')],
     ['output']),
    ('SequenceRange',
     [Array('arg0'), OptionalArray('arg1'), OptionalArray('arg2')],
     [Sequence('output')]),
    ('SequenceSeparate', [Array('input'), Int('axis')], [Sequence('output')]),
    ('SequenceUnpad', [Array('input'), Sequence('lengths')],
     [Sequence('output')]),
    ('SequenceSize', [Sequence('seq')], ['output']),
    ('SequenceLengths', [Sequence('seq')], [Sequence('output')]),
    ('SequenceCopy', [Sequence('seq')], [Sequence('output')]),
]

# Ops which modify the input in-place.
XC_SEQ_OPS_UNTYPED = [
    ('SequenceClear', [Sequence('seq')], []),
    ('SequenceAppend', [Sequence('seq'), Array('value')],
     []),
    ('SequencePop', [Sequence('seq')], [Sequence('output')]),
    ('SequenceMove', [Sequence('seq')], [Sequence('output')]),
]

XC_GENERIC_OPS = [
    ('Identity', [Array('x')], ['y']),
    ('Free', [Array('v')], []),
    ('In', [String('name')], ['v']),
    ('Out', [String('name'), Array('v')], []),
    ('Print', [ArrayList('values')], []),
    ('NullConstant', [], ['output']),

    ('GenericLen', [Array('v')], ['len']),
    ('GenericGetItem', [Array('v'), Array('index')], ['output']),
    ('GenericGetSlice',
     [Array('v'), OptionalArray('start'),
      OptionalArray('end'), OptionalArray('step')], ['output']),
    ('GenericAdd', [Array('a'), Array('b')], ['output']),
    ('GenericIs', [Array('a'), Array('b')], ['output']),
    ('GenericAccumulateGrad', [Array('a'), Array('b')], ['output']),
]


class Op(object):
    def __init__(self, name, inputs, outputs,
                 typed=True,
                 has_custom_field=False):
        self.name = name
        self.inputs = inputs
        self.outputs = []
        self.output_names = []
        for output in outputs:
            if isinstance(output, tuple):
                self.outputs.append(output)
                self.output_names.append(output[1])
            else:
                self.outputs.append(Array(output))
                self.output_names.append(output)
        self.typed = typed
        self.has_custom_field = has_custom_field


XC_ALL_OPS = [Op(*op) for op in XC_OPS]
XC_ALL_OPS += [Op(*op, has_custom_field=True) for op in XC_CUSTOM_FIELD_OPS]
XC_ALL_OPS += [Op(*op) for op in XC_SEQ_OPS]
XC_ALL_OPS += [Op(*op, typed=False) for op in XC_SEQ_OPS_UNTYPED]
XC_ALL_OPS += [Op(*op, typed=False) for op in XC_GENERIC_OPS]
