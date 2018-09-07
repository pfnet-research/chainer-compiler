import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oniku.common.codegen_util import format_code


ARRAY = 'ARRAY'
OPTIONAL_ARRAY = 'OPTIONAL_ARRAY'
ARRAY_LIST = 'ARRAY_LIST'
SEQUENCE = 'SEQUENCE'
INT = 'INT'
FLOAT = 'FLOAT'
INTS = 'INTS'
STRING = 'STRING'

XC_TYPES = [
    ARRAY, OPTIONAL_ARRAY, ARRAY_LIST, SEQUENCE, INT, FLOAT, INTS, STRING
]

STACK_VECTOR = 'chainerx::StackVector<int64_t, chainerx::kMaxNdim>'


def Array(name):
    return (ARRAY, name)


def OptionalArray(name):
    return (OPTIONAL_ARRAY, name)


def ArrayList(name):
    return (ARRAY_LIST, name)


def Sequence(name):
    return (SEQUENCE, name)


def Int(name):
    return (INT, name)


def Float(name):
    return (FLOAT, name)


def Ints(name):
    return (INTS, name)


def String(name):
    return (STRING, name)


def sigil(typ):
    if typ in [ARRAY, OPTIONAL_ARRAY]:
        return '$'
    elif typ == SEQUENCE:
        return '@'
    else:
        raise RuntimeError('Not a varaible: %s' % typ)


XC_OPS = [
    ('In', [String('name')], 'v'),
    ('Out', [String('name'), Array('v')], []),

    ('Add', [Array('a'), Array('b')], ['c']),
    ('Sub', [Array('a'), Array('b')], ['c']),
    ('Mul', [Array('a'), Array('b')], ['c']),
    ('Div', [Array('a'), Array('b')], ['c']),
    ('Pow', [Array('a'), Array('b')], ['c']),
    ('Neg', [Array('x')], ['y']),
    ('Exp', [Array('x')], ['y']),
    ('Log', [Array('x')], ['y']),
    ('Sqrt', [Array('x')], ['y']),
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
    ('Identity', [Array('x')], ['y']),
    ('Relu', [Array('x')], ['y']),
    ('ReluGrad', [Array('x'), Array('gy')], ['gx']),
    ('Shape', [Array('data')], ['shape']),
    ('Size', [Array('data')], ['size']),
    ('Reshape', [Array('data'), Array('shape')], ['reshaped']),
    ('Expand', [Array('input'), Array('shape')], ['output']),
    ('Squeeze', [Array('data'), Ints('axes')], ['squeezed']),
    ('Unsqueeze', [Array('data'), Ints('axes')], ['expanded']),
    ('Slice', [Array('data'), Ints('axes'), Ints('starts'), Ints('ends')],
     ['output']),
    ('Gather', [Array('data'), Array('indices'), Int('axis')], ['output']),
    ('SelectItem', [Array('data'), Array('indices')], ['output']),
    ('SelectItemGrad', [Array('gy'), Array('indices'), Array('shape')], ['gx']),
    ('Concat', [ArrayList('inputs'), Int('axis')], ['concat_result']),
    ('Split', [Array('input'), Int('axis'), Ints('split')], [ArrayList('outputs')]),
    ('Transpose', [Array('data'), Ints('perm')], ['transposed']),

    ('Softmax', [Array('input'), Int('axis')], ['output']),
    ('LogSoftmax', [Array('input'), Int('axis')], ['output']),

    ('MaxPool',
     [Array('x'), Ints('kernel_shape'), Ints('strides'), Ints('pads')],
     ['y']),
    ('AveragePool',
     [Array('x'), Ints('kernel_shape'), Ints('strides'), Ints('pads'),
      Int('count_include_pad')],
     ['y']),
    ('MaxPoolGrad', [Array('y'), Array('gy')], ['gx']),
    ('AveragePoolGrad', [Array('y'), Array('gy')], ['gx']),

    ('MatMul', [Array('a'), Array('b')], ['y']),
    ('Gemm',
     [Array('a'), Array('b'), Array('c'),
      Float('alpha'), Float('beta'), Int('trans_a'), Int('trans_b')],
     ['y']),
    ('LSTM',
     [Array('x'), Array('w'), Array('r'),
      OptionalArray('b'), OptionalArray('sequence_lens'),
      OptionalArray('initial_h'), OptionalArray('initial_c'),
      OptionalArray('p'),
      Int('hidden_size'),
     ],
     ['y', 'y_h', 'y_c']),

    ('BatchNormalization',
     [Array('x'), Array('s'), Array('bias'), Array('mean'), Array('var'),
      Float('epsilon'), Float('decay'), Int('spatial')],
     ['y']),
    ('BatchNormalizationGrad', [Array('y'), Array('gy')],
     ['gx0', 'gx1', 'gx2']),
    ('LRN',
     [Array('x'), Float('alpha'), Float('beta'), Float('bias'), Int('size')],
     ['y']),
    ('LRNGrad',
     [Array('x'), Array('y'), Array('gy'),
      Float('alpha'), Float('beta'), Float('bias'), Int('size')], ['gx']),

    ('Equal', [Array('a'), Array('b')], ['c']),
    ('Greater', [Array('a'), Array('b')], ['c']),
    ('GreaterEqual', [Array('a'), Array('b')], ['c']),
    ('Not', [Array('x')], ['y']),
    ('Cast', [Array('input'), Int('to')], ['output']),

    # TODO(hamaji): Re-design constants.
    ('IntConstant', [Int('value'), Int('dtype'), Int('host')], ['output']),
    ('FloatConstant', [Float('value'), Int('dtype'), Int('host')], ['output']),
    ('JmpTrue', [Array('cond'), Int('pc')], []),
    ('JmpFalse', [Array('cond'), Int('pc')], []),
]

XC_SEQ_OPS = [
    ('SequenceCreate', [], [Sequence('output')]),
    ('SequenceClear', [Sequence('seq')], []),
    ('SequenceAppend', [Sequence('seq'), Array('value')], []),
    ('SequenceLookup', [Sequence('seq'), Array('index')], ['output']),
    ('SequenceStack', [Sequence('seq'), Int('axis')], ['output']),
    ('SequencePad', [Sequence('seq'), Int('length'), Float('padding')],
     ['output']),
    ('SequenceSplit', [Array('input'), Int('axis')], [Sequence('output')]),
    ('SequenceUnpad', [Array('input'), Sequence('lengths')],
     [Sequence('output')]),
    ('SequenceSize', [Sequence('seq')], ['output']),
    ('SequenceLengths', [Sequence('seq')], [Sequence('output')]),
    ('SequenceCopy', [Sequence('seq')], [Sequence('output')]),
    ('SequenceMove', [Sequence('seq')], [Sequence('output')]),
]

XC_GENERIC_OPS = [
    ('Free', [Array('v')], []),
]


class Op(object):
    def __init__(self, name, inputs, outputs, array_only=True):
        self.name = name
        self.inputs = inputs
        self.outputs_typed = []
        self.outputs = []
        for output in outputs:
            if isinstance(output, tuple):
                self.outputs_typed.append(output)
                self.outputs.append(output[1])
            else:
                self.outputs_typed.append((ARRAY, output))
                self.outputs.append(output)
        self.array_only = array_only


XC_ALL_OPS = [Op(*op) for op in XC_OPS]
XC_ALL_OPS += [Op(*op, array_only=False) for op in XC_SEQ_OPS]
XC_ALL_OPS += [Op(*op, array_only=False) for op in XC_GENERIC_OPS]


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
    lines.append('}')

    lines.append('message XCProgramProto {')
    lines.append('repeated XCInstructionProto instructions = 1;')
    lines.append('}')

    with open('xcvm.proto', 'w') as f:
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
        if op.array_only:
            for typ, name in op.inputs:
                if typ == ARRAY:
                    args.append(f'const chainerx::Array& {name}')
                elif typ == OPTIONAL_ARRAY:
                    args.append(
                        f'const nonstd::optional<chainerx::Array>& {name}')
                elif typ == ARRAY_LIST:
                    args.append(f'const std::vector<chainerx::Array>& {name}')
            rettype = 'void'
            num_outputs = len(op.outputs)
            if num_outputs == 1:
                if op.outputs_typed[0][0] == ARRAY_LIST:
                    rettype = 'std::vector<chainerx::Array>'
                else:
                    rettype = 'chainerx::Array'
            elif num_outputs > 1:
                rettype = ('std::tuple<' +
                           ', '.join(['chainerx::Array'] * num_outputs) + '>')
        else:
            rettype = 'void'
        lines.append('%s RunImpl(%s);' % (rettype, ', '.join(args)))
        lines.append('virtual void Run(XCVMState* st);')

        lines.append('private:')
        for typ, name in op.inputs:
            ctype = None
            if typ in [ARRAY, OPTIONAL_ARRAY, INT, SEQUENCE]:
                ctype = 'int'
            elif typ == FLOAT:
                ctype = 'float'
            elif typ == STRING:
                ctype = 'std::string'
            elif typ == INTS:
                ctype = STACK_VECTOR
            elif typ == ARRAY_LIST:
                ctype = 'std::vector<int>'
            else:
                raise RuntimeError('Unknown type: %s' % typ)
            lines.append(f'{ctype} {name};')

        for typ, name in op.outputs_typed:
            if typ == ARRAY_LIST:
                lines.append('std::vector<int> %s;' % name)
            else:
                lines.append('int %s;' % name)

        lines.append('};')

    with open('gen_xcvm_ops.h', 'w') as f:
        f.write(r'''// Auto-generated by gen_xcvm.py

#pragma once

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
        for i, (typ, name) in enumerate(op.inputs):
            enum = typ.replace('OPTIONAL_', '')
            lines.append(f'CHECK_EQ(XCValueProto::{enum}, ' +
                         f'inst.inputs({i}).type()) ' +
                         f'<< "Unexpected type for input#{i} of {op.name}";')
            if typ == ARRAY or typ == OPTIONAL_ARRAY:
                lines.append('%s = inst.inputs(%d).array();' % (name, i))
            elif typ == SEQUENCE:
                lines.append('%s = inst.inputs(%d).sequence();' % (name, i))
            elif typ == INT:
                lines.append('%s = inst.inputs(%d).i();' % (name, i))
            elif typ == FLOAT:
                lines.append('%s = inst.inputs(%d).f();' % (name, i))
            elif typ == STRING:
                lines.append('%s = inst.inputs(%d).s();' % (name, i))
            elif typ == INTS:
                lines.append(f'{name} = {STACK_VECTOR}(' +
                             f'inst.inputs({i}).ints().begin(), ' +
                             f'inst.inputs({i}).ints().end());')
            elif typ == ARRAY_LIST:
                lines.append('%s.assign(inst.inputs(%d).array_list().begin(),'
                             'inst.inputs(%d).array_list().end());' %
                             (name, i, i))
            else:
                raise RuntimeError('Unknown type: %s' % typ)

        for i, (typ, name) in enumerate(op.outputs_typed):
            if typ == ARRAY_LIST:
                lines.append('%s.assign(inst.outputs().begin(), '
                             'inst.outputs().end());' % name)
            else:
                lines.append('%s = inst.outputs(%d);' % (name, i))

        lines.append('}')

        # Emit Run.
        lines.append('void %sOp::Run(XCVMState* st) {' % op.name)

        lines.append('if (st->trace_level() && !debug_info_.empty()) '
                     'std::cerr << "# " << debug_info_ << std::endl;')

        line = 'if (st->trace_level()) std::cerr'
        if op.outputs:
            for typ, name in op.outputs_typed:
                if typ == ARRAY_LIST:
                    line += f' << ArrayListToString({name})'
                else:
                    line += f' << "{sigil(typ)}" << {name}'
            line += ' << " = "'
        line += f' << "{op.name}("'
        for i, (typ, name) in enumerate(op.inputs):
            if i:
                line += ' << ", "'
            if typ in [ARRAY, OPTIONAL_ARRAY, SEQUENCE]:
                line += f' << "{sigil(typ)}" << {name}'
            elif typ in (INT, FLOAT):
                line += f' << {name}'
            elif typ == STRING:
                line += f' << "{name}"'
            elif typ == INTS:
                line += f' << StackVectorToString({name})'
            elif typ == ARRAY_LIST:
                line += f' << ArrayListToString({name})'
            else:
                raise RuntimeError('Unknown type: %s' % typ)
        line += ' << ")"'
        line += ' << std::endl;'
        lines.append(line)

        line = 'if (st->trace_level()) std::cerr'
        for typ, name in op.inputs:
            if typ in [ARRAY, OPTIONAL_ARRAY, SEQUENCE]:
                line += f' << " {sigil(typ)}" << {name} << "="'
                line += f' << st->GetVarString({name})'
        if op.outputs:
            line += ' << " ->"'
        line += ';'
        lines.append(line)

        if op.array_only:
            args = ['st']
            for typ, name in op.inputs:
                if typ == ARRAY:
                    args.append(f'st->GetVar({name})')
                elif typ == OPTIONAL_ARRAY:
                    args.append(f'st->GetVarOptional({name})')
                elif typ == ARRAY_LIST:
                    args.append(f'st->GetVarList({name})')
            call = 'RunImpl(%s)' % ', '.join(args)
            if len(op.outputs) == 1:
                if op.outputs_typed[0][0] == ARRAY_LIST:
                    lines.append('st->SetVarList(%s, %s);' %
                                 (op.outputs[0], call))
                else:
                    lines.append('st->SetVar(%s, %s);' % (op.outputs[0], call))
            elif op.outputs:
                lines.append('auto r_ = ' + call + ';')
                for i, output in enumerate(op.outputs):
                    # TODO(hamaji): Revisit optional outputs.
                    lines.append(f'if ({output} >= 0) st->SetVar({output}, std::get<{i}>(r_));')
            else:
                lines.append(call + ';')
        else:
            lines.append('RunImpl(st);')

        line = 'if (st->trace_level()) std::cerr'
        for typ, name in op.outputs_typed:
            if typ in [ARRAY, OPTIONAL_ARRAY, SEQUENCE]:
                line += f' << " {sigil(typ)}" << {name} << "="'
                line += f' << st->GetVarString({name})'
            elif typ == ARRAY_LIST:
                # TODO(hamaji): Show debug outputs of array lists.
                pass
            else:
                raise RuntimeError('Unknown output type: %s' % typ)
        line += ' << std::endl;'
        lines.append(line)

        if op.outputs:
            inputs_str = ', '.join([name for typ, name in op.inputs
                                    if typ == ARRAY or typ == OPTIONAL_ARRAY])
            outputs_str = ', '.join(op.outputs)
            lines.append('if (st->check_infs()) st->CheckInfs({%s}, {%s});' %
                         (inputs_str, outputs_str))

        lines.append('}')

    lines.append('XCVMOp* MakeXCVMOp(const XCInstructionProto& inst) {')
    lines.append('switch (inst.op()) {')
    for op in XC_ALL_OPS:
        lines.append(f'case XCInstructionProto::{op.name}:')
        lines.append(f'return new {op.name}Op(inst);')
    lines.append('default:')
    lines.append('CHECK(false) << "Unknown op: " ' +
                 '<< static_cast<int>(inst.op());')
    lines.append('}')
    lines.append('}')

    with open('gen_xcvm_ops.cc', 'w') as f:
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
    for typ, name in outputs:
        if typ == ARRAY_LIST:
            args.append(f'std::vector<int> {name}')
        else:
            args.append(f'int {name}')
    for typ, name in inputs:
        if typ in [ARRAY, OPTIONAL_ARRAY, INT, SEQUENCE]:
            args.append(f'int {name}')
        elif typ == FLOAT:
            args.append(f'float {name}')
        elif typ == STRING:
            args.append(f'const std::string& {name}')
        elif typ == INTS or typ == ARRAY_LIST:
            args.append(f'const std::vector<int>& {name}')
        else:
            raise RuntimeError('Unknown type: %s' % typ)
    args = ', '.join(args)
    return f'void Add{op}Op({args})'


def gen_xcvm_proto_util_h():
    lines = []
    for op in XC_ALL_OPS:
        signature = make_proto_signature(op.name, op.inputs, op.outputs_typed)
        lines.append(signature + ';')

    with open('xcvm_proto_util.h', 'w') as f:
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


def gen_xcvm_proto_util_cc():
    lines = []
    for op in XC_ALL_OPS:
        signature = make_proto_signature(op.name, op.inputs, op.outputs_typed)
        lines.append(signature + ' {')

        lines.append('XCInstructionProto* inst = program->add_instructions();')
        lines.append(f'inst->set_op(XCInstructionProto::{op.name});')

        for typ, name in op.inputs:
            lines.append('{')
            lines.append('XCValueProto* input_proto = inst->add_inputs();')
            enum = typ.replace('OPTIONAL_', '')
            lines.append(f'input_proto->set_type(XCValueProto::{enum});')
            if typ == ARRAY or typ == OPTIONAL_ARRAY:
                lines.append(f'input_proto->set_array({name});')
            elif typ == SEQUENCE:
                lines.append(f'input_proto->set_sequence({name});')
            elif typ == INT:
                lines.append(f'input_proto->set_i({name});')
            elif typ == FLOAT:
                lines.append(f'input_proto->set_f({name});')
            elif typ == STRING:
                lines.append(f'input_proto->set_s({name});')
            elif typ == INTS:
                lines.append(f'for (int v : {name}) input_proto->add_ints(v);')
            elif typ == ARRAY_LIST:
                lines.append(f'for (int v : {name}) '
                             'input_proto->add_array_list(v);')
            else:
                raise RuntimeError('Unknown type: %s' % typ)
            lines.append('}')

        for typ, name in op.outputs_typed:
            if typ == ARRAY_LIST:
                lines.append(f'for (int a : {name}) inst->add_outputs(a);')
            else:
                lines.append(f'inst->add_outputs({name});')

        lines.append('}')

    with open('xcvm_proto_util.cc', 'w') as f:
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
