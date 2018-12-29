import collections
import os
import subprocess
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oniku.common.codegen_util import format_code

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, help="")
args = parser.parse_args()

output_dir = args.output_dir

ARRAY = 'ARRAY'
OPTIONAL_ARRAY = 'OPTIONAL_ARRAY'
ARRAY_LIST = 'ARRAY_LIST'
SEQUENCE = 'SEQUENCE'
OPAQUE = 'OPAQUE'
INT = 'INT'
FLOAT = 'FLOAT'
INTS = 'INTS'
STRING = 'STRING'
LONGS = 'LONGS'
DOUBLES = 'DOUBLES'

ARG_TYPES = [
    ARRAY, OPTIONAL_ARRAY, ARRAY_LIST, SEQUENCE, OPAQUE
]

FIELD_TYPES = [
    INT, FLOAT, INTS, STRING, LONGS, DOUBLES
]

XC_TYPES = ARG_TYPES + FIELD_TYPES

STACK_VECTOR = 'chainerx::StackVector<int64_t, chainerx::kMaxNdim>'


_ValueInfo = collections.namedtuple('_ValueInfo', ('typ', 'name'))


class ValueInfo(_ValueInfo):
    def is_repeated(self):
        return self.typ in [INTS, ARRAY_LIST, LONGS, DOUBLES]

    def c_type(self):
        if self.typ in [ARRAY, OPTIONAL_ARRAY, INT, SEQUENCE, OPAQUE]:
            return 'int'
        elif self.typ == FLOAT:
            return 'float'
        elif self.typ == STRING:
            return 'std::string'
        elif self.typ in [INTS, ARRAY_LIST]:
            return 'std::vector<int>'
        elif self.typ == LONGS:
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
        elif self.typ == LONGS:
            return 'longs'
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


def Longs(name):
    return ValueInfo(LONGS, name)


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
    ('Tanh', [Array('x')], ['y']),
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
     [Array('data'), Array('starts'), Array('ends'), OptionalArray('axes')],
     ['output']),
    ('DynamicSliceGrad',
     [Array('gy'), Array('shape'), Array('starts'), Array('ends'),
      OptionalArray('axes')],
     ['gx']),
    ('Gather', [Array('data'), Array('indices'), Int('axis')], ['output']),
    ('GatherGrad',
     [Array('gy'), Array('indices'), Array('shape'), Int('axis')], ['gx']),
    ('SelectItem', [Array('data'), Array('indices')], ['output']),
    ('SelectItemGrad', [Array('gy'), Array('indices'), Array('shape')], ['gx']),
    ('Concat', [ArrayList('inputs'), Int('axis')], ['concat_result']),
    ('Split', [Array('input'), Int('axis'), Ints('split')],
     [ArrayList('outputs')]),
    ('Transpose', [Array('data'), Ints('perm')], ['transposed']),

    ('Softmax', [Array('input'), Int('axis')], ['output']),
    ('LogSoftmax', [Array('input'), Int('axis')], ['output']),

    ('Dropout', [Array('data'), Float('ratio')], ['output', 'mask']),

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
      Float('epsilon'), Float('decay'), Int('spatial')],
     ['y', Opaque('ctx'),
      OptionalArray('running_mean'), OptionalArray('running_var'),
      OptionalArray('saved_mean'), OptionalArray('saved_var')]),
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
     [Longs('value'), Int('dtype'), Ints('shape'), Int('host')], ['output']),
    ('FloatConstant',
     [Doubles('value'), Int('dtype'), Ints('shape'), Int('host')], ['output']),
    ('ConstantFill',
     [OptionalArray('input'), Int('dtype'), Ints('extra_shape'),
      Ints('shape'), Float('value')],
     ['output']),
    ('OneHot',
     [Array('indices'), Array('depth'), Array('values'), Int('axis')],
     ['output']),

    ('Jmp', [Int('pc')], []),
    ('JmpTrue', [Array('cond'), Int('pc')], []),
    ('JmpFalse', [Array('cond'), Int('pc')], []),

    ('ElementWiseNvrtc',
     [ArrayList('inputs'), Int('num_outputs'),
      String('code'), Int('fusion_id')],
     [ArrayList('outputs')]),
]

XC_CUSTOM_FIELD_OPS = [
    ('TVM',
     [ArrayList('inputs'), Int('num_outputs'),
      String('dso_filename'), Ints('output_shape')],
     [ArrayList('outputs')]),
]

XC_SEQ_OPS = [
    ('SequenceCreate', [], [Sequence('output')]),
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


def gen_xcvm_proto():
    lines = []
    lines.append('message XCValueProto {')
    lines.append('enum Type {')
    for i, typ in enumerate(XC_TYPES):
        lines.append('%s = %d;' % (typ, i + 1))
    lines.append('}')
    lines.append('required Type type = 1;')
    lines.append('optional int32 array = 2;')
    lines.append('optional int64 i = 3;')
    lines.append('optional double f = 4;')
    lines.append('repeated int32 ints = 5;')
    lines.append('optional string s = 6;')
    lines.append('repeated int32 array_list = 7;')
    lines.append('optional int32 sequence = 8;')
    lines.append('repeated int64 longs = 9;')
    lines.append('repeated double doubles = 10;')
    lines.append('optional int32 opaque = 11;')
    lines.append('}')

    lines.append('message XCInstructionProto {')
    lines.append('enum Op {')
    for i, op in enumerate(XC_ALL_OPS):
        lines.append('%s = %d;' % (op.name, i + 1))
    lines.append('}')
    lines.append('required Op op = 1;')
    lines.append('repeated XCValueProto inputs = 2;')
    lines.append('repeated int32 outputs = 3;')
    lines.append('optional string debug_info = 4;')
    lines.append('optional int64 id = 5;')
    lines.append('}')

    lines.append('message XCProgramProto {')
    lines.append('repeated XCInstructionProto instructions = 1;')
    lines.append('}')

    with open(output_dir + '/xcvm.proto', 'w') as f:
        f.write(r'''// Auto-generated by gen_xcvm.py

syntax = "proto2";

package oniku.runtime;

''')
        f.writelines(format_code(lines))

    subprocess.check_call(['protoc', 'xcvm.proto', '--cpp_out=.'])


def gen_gen_xcvm_ops_h():
    lines = []

    for op in XC_ALL_OPS:
        lines.append('class %sOp : public XCVMOp {' % op.name)
        lines.append('public:')
        lines.append('explicit %sOp(const XCInstructionProto& inst);' % op.name)

        args = ['XCVMState* st']
        if op.typed:
            for typ, name in op.inputs:
                if typ == ARRAY:
                    args.append('const chainerx::Array& %s' % name)
                elif typ == OPTIONAL_ARRAY:
                    args.append(
                        'const nonstd::optional<chainerx::Array>& %s' % name)
                elif typ == ARRAY_LIST:
                    args.append('const std::vector<chainerx::Array>& %s' % name)
                elif typ == SEQUENCE:
                    args.append('const XCVMSequence& %s' % name)
                elif typ == OPAQUE:
                    args.append('const XCVMOpaque& %s' % name)
                else:
                    assert typ in FIELD_TYPES, 'Unknown type: %s' % typ

            output_ctypes = []
            for typ, name in op.outputs:
                if typ == ARRAY_LIST:
                    output_ctypes.append('std::vector<chainerx::Array>')
                elif typ == SEQUENCE:
                    args.append('XCVMSequence* %s' % name)
                elif typ == OPAQUE:
                    output_ctypes.append('XCVMOpaque*')
                else:
                    output_ctypes.append('chainerx::Array')

            if len(output_ctypes) == 0:
                rettype = 'void'
            elif len(output_ctypes) == 1:
                rettype = output_ctypes[0]
            else:
                rettype = 'std::tuple<' + ', '.join(output_ctypes) + '>'
        else:
            rettype = 'void'
        lines.append('%s RunImpl(%s);' % (rettype, ', '.join(args)))
        lines.append('virtual void Run(XCVMState* st);')

        lines.append('private:')
        for inp in op.inputs:
            ctype = inp.c_storage_type()
            lines.append('%s %s;' % (ctype, inp.name))

        for out in op.outputs:
            ctype = out.c_storage_type()
            lines.append('%s %s;' % (ctype, out.name))

        if op.has_custom_field:
            lines.append('~%sOp() override;' % op.name)
            lines.append('void InitImpl();')
            lines.append('class %sImpl;' % op.name)
            lines.append('%sImpl* impl_{nullptr};' % op.name)

        lines.append('};')

    with open(output_dir + '/gen_xcvm_ops.h', 'w') as f:
        f.write(r'''// Auto-generated by gen_xcvm.py

#pragma once

#include <memory>
#include <string>

#include <chainerx/stack_vector.h>

#include <runtime/xcvm_op.h>
#include <runtime/xcvm_state.h>
#include <runtime/xcvm.pb.h>

namespace oniku {
namespace runtime {

''')
        f.writelines(format_code(lines))
        f.write(r'''
}  // namespace runtime
}  // namespace oniku
''')


def gen_gen_xcvm_ops_cc():
    lines = []

    for op in XC_ALL_OPS:
        # Emit constructor.
        lines.append('%sOp::%sOp(const XCInstructionProto& inst) {' %
                     (op.name, op.name))
        for i, inp in enumerate(op.inputs):
            enum = inp.typ.replace('OPTIONAL_', '')
            lines.append('CHECK_EQ(XCValueProto::%s, ' % (enum) +
                         'inst.inputs(%d).type()) ' % (i) +
                         '<< "Unexpected type for input#%d of %s";' % (i, op.name))
            pfn = inp.proto_field_name()
            name = inp.name
            if not inp.is_repeated():
                lines.append('%s = inst.inputs(%d).%s();' % (name, i, pfn))
            elif inp.typ == INTS:
                lines.append('%s = %s(' % (name, STACK_VECTOR) +
                             'inst.inputs(%d).ints().begin(), ' % (i) +
                             'inst.inputs(%d).ints().end());' % (i))
            else:
                lines.append('%s.assign(inst.inputs(%d).%s().begin(),' % (name, i, pfn) +
                             'inst.inputs(%d).%s().end());' % (i, pfn))

        for i, (typ, name) in enumerate(op.outputs):
            if typ == ARRAY_LIST:
                lines.append('%s.assign(inst.outputs().begin(), '
                             'inst.outputs().end());' % name)
            else:
                lines.append('%s = inst.outputs(%d);' % (name, i))

        if op.has_custom_field:
            lines.append('InitImpl();')

        lines.append('}')

        if op.has_custom_field:
            lines.append('%sOp::~%sOp() {' % (op.name, op.name))
            lines.append('delete impl_;')
            lines.append('}')

        # Emit Run.
        lines.append('void %sOp::Run(XCVMState* st) {' % op.name)

        lines.append('if (st->trace_level() && !debug_info_.empty()) '
                     'std::cerr << "# " << debug_info_ << std::endl;')

        line = 'if (st->trace_level()) std::cerr'
        if op.outputs:
            for typ, name in op.outputs:
                if typ == ARRAY_LIST:
                    line += ' << ArrayListToString(%s)' % name
                else:
                    line += ' << "%s" << %s' % (sigil(typ), name)
            line += ' << " = "'
        line += ' << "%s("' % (op.name)
        for i, (typ, name) in enumerate(op.inputs):
            if i:
                line += ' << ", "'
            if typ in [ARRAY, OPTIONAL_ARRAY, SEQUENCE, OPAQUE]:
                line += ' << "%s" << %s' % (sigil(typ), name)
            elif typ in (INT, FLOAT):
                line += ' << %s' % name
            elif typ in [STRING, LONGS, DOUBLES]:
                line += ' << "%s"' % name
            elif typ == INTS:
                line += ' << StackVectorToString(%s)' % name
            elif typ == ARRAY_LIST:
                line += ' << ArrayListToString(%s)' % name
            else:
                raise RuntimeError('Unknown type: %s' % typ)
        line += ' << ")"'
        line += ' << std::endl;'
        lines.append(line)

        line = 'if (st->trace_level()) std::cerr'
        for typ, name in op.inputs:
            if typ in [ARRAY, OPTIONAL_ARRAY, SEQUENCE]:
                line += ' << " %s" << %s << "="' % (sigil(typ), name)
                line += ' << st->GetVarString(%s)' % name
            elif typ == ARRAY_LIST:
                line += ' << st->GetVarListString(%s)' % name
        if op.outputs:
            line += ' << " ->"'
        if not line.endswith('std::cerr'):
            line += ';'
            lines.append(line)

        if op.typed:
            args = ['st']

            # TODO(hamaji): Remove this code by removing null gradients.
            conds = []
            for typ, name in op.inputs:
                if typ in ARG_TYPES and typ != ARRAY_LIST:
                    conds.append('(%s >= 0 && st->GetVar(%s)->IsNull())' %
                                 (name, name))
            if conds:
                lines.append('if (%s) {' % (' || '.join(conds)))
                lines.append('WARN_ONCE("%s skipped\\n");' % op.name)
                for typ, oname in op.outputs:
                    if typ in ARG_TYPES and typ != ARRAY_LIST:
                        lines.append('st->SetVar(%s, XCVMVar());' % oname)
                lines.append('return;')
                lines.append('}')

            for typ, name in op.inputs:
                if typ == ARRAY:
                    args.append('st->GetArray(%s)' % name)
                elif typ == OPTIONAL_ARRAY:
                    args.append('st->GetOptionalArray(%s)' % name)
                elif typ == ARRAY_LIST:
                    args.append('st->GetArrayList(%s)' % name)
                elif typ == SEQUENCE:
                    args.append('*st->GetSequence(%s)' % name)
                elif typ == OPAQUE:
                    args.append('st->GetOpaque(%s)' % name)

            outputs = []
            for output in op.outputs:
                typ, name = output
                if typ == SEQUENCE:
                    args.append('st->CreateSequence(%s)' % name)
                else:
                    outputs.append(output)

            call = 'RunImpl(%s)' % ', '.join(args)
            if len(outputs) == 1:
                typ, name = outputs[0]
                if typ == ARRAY_LIST:
                    lines.append('st->SetArrayList(%s, %s);' % (name, call))
                elif typ == OPAQUE:
                    lines.append('st->SetOpaque(%s, %s);' % (name, call))
                else:
                    lines.append('st->SetArray(%s, %s);' % (name, call))
            elif outputs:
                lines.append('auto r_ = ' + call + ';')
                for i, (typ, output) in enumerate(outputs):
                    # TODO(hamaji): Revisit optional outputs.
                    line = 'if (%s >= 0) ' % output
                    if typ == OPAQUE:
                        line += 'st->SetOpaque(%s, std::get<%d>(r_));' % (output, i)
                    else:
                        line += 'st->SetArray(%s, std::get<%d>(r_));' % (output, i)
                    lines.append(line)
            else:
                lines.append(call + ';')
        else:
            lines.append('RunImpl(st);')

        line = 'if (st->trace_level()) std::cerr'
        for typ, name in op.outputs:
            if typ in [ARRAY, OPTIONAL_ARRAY, SEQUENCE, OPAQUE]:
                line += ' << " %s" << %s << "="' % (sigil(typ), name)
                line += ' << st->GetVarString(%s)' % name
            elif typ == ARRAY_LIST:
                line += ' << st->GetVarListString(%s)' % name
            else:
                raise RuntimeError('Unknown output type: %s' % typ)
        line += ' << std::endl;'
        lines.append(line)

        if op.outputs:
            inputs_str = ', '.join([name for typ, name in op.inputs
                                    if typ == ARRAY or typ == OPTIONAL_ARRAY])
            outputs_str = ', '.join(op.output_names)
            lines.append('if (st->check_infs()) st->CheckInfs({%s}, {%s});' %
                         (inputs_str, outputs_str))
            lines.append('if (st->check_nans()) st->CheckNans({%s}, {%s});' %
                         (inputs_str, outputs_str))

        lines.append('}')

    lines.append('XCVMOp* MakeXCVMOp(const XCInstructionProto& inst) {')
    lines.append('switch (inst.op()) {')
    for op in XC_ALL_OPS:
        lines.append('case XCInstructionProto::%s:' % (op.name))
        lines.append('return new %sOp(inst);' % (op.name))
    lines.append('default:')
    lines.append('CHECK(false) << "Unknown op: " ' +
                 '<< static_cast<int>(inst.op());')
    lines.append('}')
    lines.append('}')

    with open(output_dir + '/gen_xcvm_ops.cc', 'w') as f:
        f.write(r'''// Auto-generated by gen_xcvm.py

#include <string>
#include <sstream>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>

namespace oniku {
namespace runtime {

std::string StackVectorToString(const chainerx::StackVector<int64_t, chainerx::kMaxNdim>& s) {
    std::ostringstream oss;
    for (int v : s) {
        oss << (oss.str().empty() ? '(' : ',');
        oss << v;
    }
    oss << ')';
    return oss.str();
}

std::string ArrayListToString(const std::vector<int>& s) {
    std::ostringstream oss;
    for (int v : s) {
        oss << (oss.str().empty() ? '(' : ',');
        oss << '$' << v;
    }
    oss << ')';
    return oss.str();
}

''')
        f.writelines(format_code(lines))
        f.write(r'''
}  // namespace runtime
}  // namespace oniku
''')


def make_proto_signature(op, inputs, outputs):
    args = ['XCProgramProto* program']
    for out in outputs:
        args.append('%s %s' % (out.c_type(), out.name))
    for inp in inputs:
        args.append('%s %s' % (inp.c_type(), inp.name))
    args = ', '.join(args)
    return 'void Add%sOp(%s)' % (op, args)


def gen_xcvm_proto_util_h():
    lines = []
    for op in XC_ALL_OPS:
        signature = make_proto_signature(op.name, op.inputs, op.outputs)
        lines.append(signature + ';')

    with open(output_dir + '/xcvm_proto_util.h', 'w') as f:
        f.write(r'''// Auto-generated by gen_xcvm.py

#pragma once

#include <runtime/xcvm.pb.h>

namespace oniku {
namespace runtime {

''')
        f.writelines(format_code(lines))
        f.write(r'''
}  // namespace runtime
}  // namespace oniku
''')


def gen_xcvm_proto_util_cc():
    lines = []
    for op in XC_ALL_OPS:
        signature = make_proto_signature(op.name, op.inputs, op.outputs)
        lines.append(signature + ' {')

        lines.append('XCInstructionProto* inst = program->add_instructions();')
        lines.append('inst->set_op(XCInstructionProto::%s);' % op.name)

        for inp in op.inputs:
            lines.append('{')
            lines.append('XCValueProto* input_proto = inst->add_inputs();')
            enum = inp.typ.replace('OPTIONAL_', '')
            lines.append('input_proto->set_type(XCValueProto::%s);' % enum)
            pfn = inp.proto_field_name()
            name = inp.name
            if inp.is_repeated():
                lines.append('for (auto v : %s) ' % name +
                             'input_proto->add_%s(v);' % pfn)
            else:
                lines.append('input_proto->set_%s(%s);' % (pfn, name))
            lines.append('}')

        for typ, name in op.outputs:
            if typ == ARRAY_LIST:
                lines.append('for (int a : %s) inst->add_outputs(a);' % name)
            else:
                lines.append('inst->add_outputs(%s);' % name)

        lines.append('}')

    with open(output_dir + '/xcvm_proto_util.cc', 'w') as f:
        f.write(r'''// Auto-generated by gen_xcvm.py

#include <runtime/xcvm.pb.h>

namespace oniku {
namespace runtime {

''')
        f.writelines(format_code(lines))
        f.write(r'''
}  // namespace runtime
}  // namespace oniku
''')


gen_xcvm_proto()
gen_gen_xcvm_ops_h()
gen_gen_xcvm_ops_cc()
gen_xcvm_proto_util_h()
gen_xcvm_proto_util_cc()
