"""Generates boilerplate code for Oniku's Node class.

Nodes in ONNX are very flexible. They allow arbitrary strings as their
operation type (e.g., "Conv") and attribute keys (e.g., "pads"). As we
would limit and expand pre-defined sets of ONNX operations and
attributes, this file will be the definition of what are supported.
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oniku.common import codegen_util


def attr_sets(**kwargs):
    return kwargs


class Required(object):
    def __init__(self, v):
        self.v = v


class Tensor():
    pass


class Graph():
    pass


class Dtype(object):
    pass


ONIKUX_GLOBAL_ATTRS = attr_sets(onikux_order=-1, onikux_fusion_group=0)

NODES = []


class NodeDef(object):

    def __init__(self, op_type, num_inputs, num_outputs, **kwargs):
        self.op_type = op_type
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.attributes = kwargs
        self.attributes.update(ONIKUX_GLOBAL_ATTRS)
        self.attr_defs = {}  # To be filled after parsed.
        NODES.append(self)


NodeDef('Identity', 1, 1)
NodeDef('Neg', 1, 1)
NodeDef('Reciprocal', 1, 1)
NodeDef('Exp', 1, 1)
NodeDef('Log', 1, 1)
NodeDef('Sqrt', 1, 1)
NodeDef('Tanh', 1, 1)
NodeDef('Abs', 1, 1)
NodeDef('Relu', 1, 1)
NodeDef('Selu', 1, 1,
        alpha=1.67326319217681884765625, gamma=1.05070102214813232421875)
NodeDef('LeakyRelu', 1, 1, alpha=0.01)
NodeDef('Elu', 1, 1, alpha=1.0)
NodeDef('Sigmoid', 1, 1)
NodeDef('Not', 1, 1)
NodeDef('Floor', 1, 1)
NodeDef('Ceil', 1, 1)
NodeDef('Softplus', 1, 1)
NodeDef('Softsign', 1, 1)

NodeDef('Add', 2, 1)
NodeDef('Sub', 2, 1)
NodeDef('Mul', 2, 1)
NodeDef('Div', 2, 1)
NodeDef('Pow', 2, 1)
NodeDef('Equal', 2, 1)
NodeDef('Greater', 2, 1)
NodeDef('Less', 2, 1)

NodeDef('Constant', 0, 1, tensor_value=Required(Tensor), onikux_host=False)
NodeDef('ConstantLike', (0, 1), 1,
        dtype=Dtype, shape=[int], value=0.0)
NodeDef('ConstantFill', (0, 1), 1,
        dtype=Dtype, extra_shape=[int], input_as_shape=int,
        shape=[int], value=0.0)
NodeDef('OneHot', 3, 1, axis=-1)
NodeDef('Cast', 1, 1, to=Required(Dtype))
NodeDef('Shape', 1, 1)
NodeDef('Size', 1, 1)
NodeDef('Reshape', 2, 1)
NodeDef('Expand', 2, 1)
NodeDef('Squeeze', 1, 1, axes=[int])
NodeDef('Unsqueeze', 1, 1, axes=Required([int]))
NodeDef('Flatten', 1, 1, axis=1)
NodeDef('Slice', 1, 1, axes=[int], starts=[int], ends=[int])
NodeDef('DynamicSlice', (3, 4), 1)
NodeDef('Gather', 2, 1, axis=0)
NodeDef('Concat', None, 1, axis=Required(int))
NodeDef('Split', 1, None, axis=0, split=[int])
NodeDef('Transpose', 1, 1, perm=[int])

NodeDef('Sum', None, 1)
NodeDef('Mean', None, 1)
NodeDef('Max', None, 1)
NodeDef('Min', None, 1)
NodeDef('Clip', 1, 1, max=float('inf'), min=float('-inf'))

NodeDef('ReduceSum', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceSumSquare', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceMean', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceMax', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceMin', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceL1', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceL2', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceLogSum', 1, 1, axes=[int], keepdims=True)
NodeDef('ReduceLogSumExp', 1, 1, axes=[int], keepdims=True)

NodeDef('ArgMax', 1, 1, axis=0, keepdims=True)
NodeDef('ArgMin', 1, 1, axis=0, keepdims=True)
NodeDef('Hardmax', 1, 1, axis=1)

NodeDef('Dropout', 1, (1, 2), ratio=0.5)

NodeDef('MatMul', 2, 1)
NodeDef('Gemm', 3, 1, alpha=1.0, beta=1.0, transA=False, transB=False)

NodeDef('RNN', (3, 4, 5, 6), (0, 1, 2),
        activation_alpha=[float], activation_beta=[float],
        activations=[str], clip=float, direction='forward',
        hidden_size=int)
NodeDef('GRU', (3, 4, 5, 6), (0, 1, 2),
        activation_alpha=[float], activation_beta=[float],
        activations=[str], clip=float, direction='forward',
        hidden_size=int, linear_before_reset=0)
# Extension: The fouth output is for backward context.
NodeDef('LSTM', (3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4),
        activation_alpha=[float], activation_beta=[float],
        activations=[str], clip=float, direction='forward',
        hidden_size=int, input_forget=0)

conv_attrs = attr_sets(auto_pad='NOTSET',
                       dilations=[int],
                       group=1,
                       kernel_shape=[int],
                       pads=[int],
                       strides=[int])
NodeDef('Conv', (2, 3), 1, **conv_attrs)
NodeDef('ConvTranspose', (2, 3), 1,
        output_padding=[int], output_shape=[int], **conv_attrs)

# Extension: the second or the sixth output is for backward context.
NodeDef('BatchNormalization', 5, (1, 2, 5, 6), epsilon=1e-5, momentum=0.9, spatial=1)
# Extension: the second output is for backward context.
NodeDef('LRN', 1, (1, 2), alpha=1e-4, beta=0.75, bias=1.0, size=Required(int))

pool_attrs = attr_sets(auto_pad='NOTSET',
                       kernel_shape=Required([int]),
                       pads=[int],
                       storage_order=0,
                       strides=[int])
# Extension: the third output is for backward context.
NodeDef('MaxPool', 1, (1, 2, 3), onikux_cover_all=False, **pool_attrs)
# Extension: the second output is for backward context.
NodeDef('AveragePool', 1, (1, 2), count_include_pad=False, **pool_attrs)
NodeDef('GlobalMaxPool', 1, 1)
NodeDef('GlobalAveragePool', 1, 1)
NodeDef('Pad', 1, 1, mode='constant', pads=[int], value=0.0)

NodeDef('Softmax', 1, 1, axis=1)
NodeDef('LogSoftmax', 1, 1, axis=1)
# Extension: it takes N+1 inputs.
NodeDef('If', None, None, else_branch=Graph, then_branch=Graph)
NodeDef('Loop', None, None, body=Graph, onikux_stack_axis=0)
NodeDef('Scan', None, None, body=Graph, num_scan_inputs=Required(int))

NodeDef('OnikuxReluGrad', 2, 1)
NodeDef('OnikuxReduceSumTo', 2, 1)
NodeDef('OnikuxMaxPoolGrad', 2, 1)
NodeDef('OnikuxAveragePoolGrad', 2, 1)
NodeDef('OnikuxBatchNormalizationGrad', 2, 3)
NodeDef('OnikuxConvTransposeWithDynamicOutputShape', 3, 1, **conv_attrs)
NodeDef('OnikuxSoftmaxCrossEntropy', 2, 1)
NodeDef('OnikuxSelectItem', 2, 1)
NodeDef('OnikuxSelectItemGrad', 3, 1)
NodeDef('OnikuxLRNGrad', 4, 1,
        alpha=1e-4, beta=0.75, bias=1.0, size=Required(int))
NodeDef('OnikuxLSTMGrad', 2, 4)
NodeDef('OnikuxConvGradWeight', 3, 1, **conv_attrs)
NodeDef('OnikuxGatherGrad', 3, 1, axis=0)
# body_ref is a name of a sub Graph in a sibling Loop node.
NodeDef('OnikuxLoopRef', None, None,
        body_ref=Required(str),
        input_value_names=[str], output_value_names=[str],
        onikux_stack_axis=0)
NodeDef('OnikuxIfRef', None, None,
        then_branch_ref=Required(str),
        else_branch_ref=Required(str),
        then_input_value_names=[str], then_output_value_names=[str],
        else_input_value_names=[str], else_output_value_names=[str])
NodeDef('OnikuxDynamicSliceGrad', (4, 5), 1)
NodeDef('OnikuxFusionGroup', None, None, subgraph=Graph)

NodeDef('OnikuxBackpropStackPush', 1, 0, id=Required(int))
NodeDef('OnikuxBackpropStackPop', 0, 1, id=Required(int))

NodeDef('OnikuxPrint', None, 0)

# Creates a new sequence: () -> ([T])
NodeDef('OnikuxSequenceCreate', 0, 1)

# Appends an element to a sequence: ([T], T) -> ([T])
NodeDef('OnikuxSequenceAppend', 2, 1)

# Pops an element from a sequence: ([T]) -> ([T], T)
NodeDef('OnikuxSequencePop', 1, 2)

# Looks up an element in a sequence: ([T], I) -> (T)
NodeDef('OnikuxSequenceLookup', 2, 1)

# Equivalent to Python's __getitem__ for a slice: ([T], I, I, I) -> ([T])
NodeDef('OnikuxSequenceGetSlice', (1, 2, 3, 4), 1)

# Stacks elements in a sequence: ([T]) -> (T)
NodeDef('OnikuxSequenceStack', 1, 1, axis=0)

# Concatenates elements in a sequence: ([T]) -> (T)
# The second output is for backward context.
NodeDef('OnikuxSequenceConcat', 1, (1, 2), axis=0)

# Pads elements in a sequence: ([T]) -> (T)
NodeDef('OnikuxSequencePad', 1, 1, length=0, value=0.0)

# Splits a tensor to a sequence: (T) -> ([T])
NodeDef('OnikuxSequenceSplit', 1, 1, axis=0)

# Strips paddings in a tensor and returns a sequence: (T, [I]) -> ([T])
# Note the result of SequenceLengths can be used as the second argument.
NodeDef('OnikuxSequenceUnpad', 2, 1)

# Returns the number of elements in a sequence: ([T]) -> (I)
NodeDef('OnikuxSequenceSize', 1, 1)

# Returns lengths of elements in a sequence: ([T]) -> ([I])
NodeDef('OnikuxSequenceLengths', 1, 1)

# Equivalent to Python's range.
NodeDef('OnikuxSequenceRange', (1, 2, 3), 1)

# The gradients of sequence related ops.
NodeDef('OnikuxSequenceLookupGrad', 3, 1)
NodeDef('OnikuxSequenceGetSliceGrad', (2, 3, 4, 5), 1)
NodeDef('OnikuxSequenceConcatGrad', 2, 1, axis=0)

# Equivalent to Python's __len__.
# For tensors: Gather(Shape(input0), 0)
# For sequences: OnikuxSequenceSize(input0)
NodeDef('OnikuxGenericLen', 1, 1)

# Equivalent to Python's __getitem__ for a scalar index.
# For tensors: Gather(input0, input1)
# For sequences: OnikuxSequenceLookup(input0, input1)
# TODO(hamaji): Deprecate this op.
NodeDef('OnikuxGenericGetItem', 2, 1)

# Equivalent to Python's __getitem__ for a slice.
# For tensors: DynamicSlice(input0, [input1], [input2]) -> tensor
# For sequences: input0[input1:input2:input3] in Python -> sequence
# TODO(hamaji): Deprecate this op.
NodeDef('OnikuxGenericGetSlice', (1, 2, 3, 4), 1)

# Equivalent to Python's __add__.
# For tensors: Add(input0, input1) -> tensor
# For sequences: input0 + input1 in Python -> sequence
# TODO(hamaji): Deprecate this op.
NodeDef('OnikuxGenericAdd', 2, 1)

# Similar to Python's `is` keyword.
# This returns true only when both inputs are bool scalars and have
# the same value.
NodeDef('OnikuxGenericIs', 2, 1)

# Creates an initialized gradient value.
# For tensors: ZerosLike(input)
# For sequence: vector(input.size(), None)
NodeDef('OnikuxGenericZerosLikeGrad', 1, 1)

# Accumulates two gradient values.
# For tensors: Add(input0, input1) -> tensor
# For sequence: Add(i0, i1) for each element in sequences.
NodeDef('OnikuxGenericAccumulateGrad', 2, 1)


class AttrDef(object):
    def __init__(self, name, value):
        self.name = name
        self.onnx_name = self.name
        if self.onnx_name == 'tensor_value':
            self.onnx_name = 'value'
        self.c_name = re.sub(r'[A-Z]', lambda m: '_' + m.group(0).lower(), name)
        self.required = False
        self.value = None
        self.op_types = []
        if isinstance(value, Required):
            self.required = True
            value = value.v
        if isinstance(value, list) or isinstance(value, type):
            self.type = value
        else:
            self.type = type(value)
            self.value = value
            assert self.type in (bool, int, str, float, Dtype, Tensor, Graph)

    def c_type(self, typ=None):
        typ = self.type if typ is None else typ
        if isinstance(typ, list):
            assert len(typ) == 1
            return 'std::vector<%s>' % self.c_type(typ[0])
        return {
            bool: 'bool',
            int: 'int64_t',
            float: 'float',
            str: 'std::string',
            Dtype: 'Dtype',
            Tensor: 'std::unique_ptr<Tensor>',
            Graph: 'std::unique_ptr<Graph>',
        }[typ]

    def c_arg_type(self):
        typ = self.c_type()
        if 'std::' in typ:
            return 'const %s&' % (typ)
        return typ

    def c_setter_arg_type(self):
        if self.type == Tensor:
            return 'Tensor*'
        if self.type == Graph:
            return 'Graph*'
        return self.c_arg_type()

    def onnx_type(self, typ=None):
        typ = self.type if typ is None else typ
        if isinstance(typ, list):
            assert len(typ) == 1
            return '%sS' % self.onnx_type(typ[0])
        return {
            bool: 'onnx::AttributeProto::INT',
            int: 'onnx::AttributeProto::INT',
            float: 'onnx::AttributeProto::FLOAT',
            str: 'onnx::AttributeProto::STRING',
            Dtype: 'onnx::AttributeProto::INT',
            Tensor: 'onnx::AttributeProto::TENSOR',
            Graph: 'onnx::AttributeProto::GRAPH',
        }[typ]

    def onnx_field(self, typ=None):
        typ = self.type if typ is None else typ
        if isinstance(typ, list):
            assert len(typ) == 1
            return self.onnx_field(typ[0]) + 's'
        return {
            bool: 'int',
            int: 'int',
            float: 'float',
            str: 'string',
            Dtype: 'dtype',
            Tensor: 'tensor',
            Graph: 'graph',
        }[typ]

    def add_func(self, typ=None):
        return 'add_%s_attr' % self.onnx_field()


ATTRS = {}

for node in NODES:
    for name, value in node.attributes.items():
        attr = AttrDef(name, value)
        node.attr_defs[name] = attr
        if name in ATTRS:
            assert attr.type == ATTRS[name].type
        else:
            ATTRS[name] = attr
        ATTRS[name].op_types.append(node.op_type)


ATTRS = [a for _, a in sorted(ATTRS.items())]


def gen_gen_node_base_h():
    public_lines = []
    private_lines = []
    public_lines.append('enum OpType {')
    for node in NODES:
        public_lines.append('k%s,' % (node.op_type))
    public_lines.append('};')
    public_lines.append('static const char* OpTypeToString(OpType op_type);')
    public_lines.append('OpType op_type() const {')
    public_lines.append('return op_type_;')
    public_lines.append('}')

    for attr in ATTRS:
        name = attr.c_name
        arg = attr.c_arg_type()
        typ = attr.c_type()

        public_lines.append('%s %s() const ' % (arg, name) + '{')
        public_lines.append('return %s_;' % (name))
        public_lines.append('}')

        if attr.type == Graph:
            public_lines.append('Graph* release_%s() ' % (name) + '{')
            public_lines.append('return %s_.release();' % (name))
            public_lines.append('}')

        sarg = attr.c_setter_arg_type()
        public_lines.append('NodeBase* set_%s(%s %s);' % (name, sarg, name))
        private_lines.append('%s %s_;' % (typ, name))
        private_lines.append('bool was_%s_set_ = false;' % (name))

    lines = public_lines + ['protected:'] + private_lines

    with open('gen_node_base.h', 'w') as f:
        f.write(r'''// Auto-generated by gen_node.py

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnx/onnx_pb.h>

#include <compiler/dtype.h>

namespace oniku {

class Graph;
class Tensor;
class Value;

class NodeBase {
public:
    void FillONNXAttributes(onnx::NodeProto* xnode) const;

    void SetDefaultAttributeValues();

    void ValidateNumInputsOutputs(const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) const;
    void ValidateAttributes() const;

''')
        f.writelines(codegen_util.format_code(lines, num_indents=4))
        f.write(r'''
    OpType op_type_;
    std::vector<onnx::AttributeProto> unknown_attributes_;

    explicit NodeBase(OpType op_type);
    NodeBase(const onnx::NodeProto& xnode, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs);
};

}  // namespace oniku
''')


def gen_gen_node_base_cc():
    lines = []
    lines.append('NodeBase::NodeBase(OpType op_type) : op_type_(op_type) {}')
    lines.append('NodeBase::NodeBase(const onnx::NodeProto& xnode, '
                 'const std::vector<Value*>& inputs, '
                 'const std::vector<Value*>& outputs) {')
    for i, node in enumerate(NODES):
        lines.append('if (xnode.op_type() == "%s") ' % (node.op_type) + '{')
        if i:
            lines[-1] = '} else ' + lines[-1]
        lines.append('op_type_ = k%s;' % (node.op_type))
    lines.append('} else {')
    lines.append('CHECK(false) << "Unsupported op_type: " '
                 '<< xnode.op_type();')
    lines.append('}')

    lines.append('SetDefaultAttributeValues();')
    lines.append('ValidateNumInputsOutputs(inputs, outputs);')
    lines.append('')

    lines.append('// Validate attributes.')
    lines.append('switch (op_type_) {')
    for node in NODES:
        op = node.op_type
        lines.append('case k%s: ' % (node.op_type) + '{')
        lines.append('for (const onnx::AttributeProto& xattr : '
                     'xnode.attribute()) {')
        conds = []
        bodies = []
        for _, attr in sorted(node.attr_defs.items()):
            conds.append('xattr.name() == "%s"' % (attr.onnx_name))
            blines = []
            blines.append('if (!g_permissive) '
                          'CHECK_EQ(xattr.type(), %s);' % (attr.onnx_type()))
            if attr.type == int:
                blines.append('set_%s(xattr.i());' % (attr.c_name))
            elif attr.type == bool:
                blines.append('set_%s(xattr.i() != 0);' % (attr.c_name))
            elif attr.type == float:
                blines.append('set_%s(xattr.f());' % (attr.c_name))
            elif attr.type == str:
                blines.append('set_%s(xattr.s());' % (attr.c_name))
            elif isinstance(attr.type, list):
                fs = attr.onnx_field()
                blines.append('%s_.assign(xattr.%s().begin(), ' % (attr.c_name, fs) +
                              'xattr.%s().end());' % (fs))
            elif attr.type == Dtype:
                blines.append('set_%s(Dtype(onnx::TensorProto::DataType(xattr.i())));' % (attr.c_name))
            elif attr.type == Tensor:
                blines.append('set_%s(new Tensor(xattr.t()));' % (attr.c_name))
            elif attr.type == Graph:
                blines.append('set_%s(new Graph(xattr.g()));' % (attr.c_name))
            else:
                raise RuntimeError('Unknown attribute type: %s' % attr.type)
            blines.append('was_%s_set_ = true;' % (attr.c_name))
            bodies.append(blines)
        bodies.append([
            'if (!g_permissive) CHECK(false) << "Invalid attribute `"'
            '<< xattr.name() << "\' for " << OpTypeToString(op_type_);',
            'unknown_attributes_.push_back(xattr);'])
        lines += codegen_util.cond(conds, bodies)

        lines.append('}')
        lines.append('}')
        lines.append('break;')
        lines.append('}')

    lines.append('}')

    lines.append('ValidateAttributes();')

    lines.append('}')

    lines.append('const char* NodeBase::OpTypeToString(OpType op_type) {')
    lines.append('switch (op_type) {')
    for node in NODES:
        lines.append('case NodeBase::k%s: ' % (node.op_type) +
                     'return "%s";' % (node.op_type))
    lines.append('default: CHECK(false) << "Unknown op_type: " << '
                 'static_cast<int>(op_type);')
    lines.append('}')
    lines.append('}')

    lines.append('void NodeBase::FillONNXAttributes(onnx::NodeProto* xnode) '
                 'const {')

    lines.append(r'''
    auto add_int_attr = [&xnode](const std::string& name, int v) {
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::INT);
        xattr->set_i(v);
    };

    auto add_float_attr = [&xnode](const std::string& name, float v) {
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::FLOAT);
        xattr->set_f(v);
    };

    auto add_string_attr = [&xnode](const std::string& name, const std::string& v) {
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::STRING);
        xattr->set_s(v);
    };

    auto add_tensor_attr = [&xnode](const std::string& name, const std::unique_ptr<Tensor>& v) {
        if (!v.get()) return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::TENSOR);
        v->ToONNX(xattr->mutable_t());
    };

    auto add_graph_attr = [&xnode](const std::string& name, const std::unique_ptr<Graph>& v) {
        if (!v.get()) return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::GRAPH);
        v->ToONNX(xattr->mutable_g());
    };

    auto add_ints_attr = [&xnode](const std::string& name, const std::vector<int64_t>& ints) {
        if (ints.empty()) return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::INTS);
        for (int s : ints) xattr->add_ints(s);
    };

    auto add_floats_attr = [&xnode](const std::string& name, const std::vector<float>& floats) {
        if (floats.empty()) return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::FLOATS);
        for (float s : floats) xattr->add_floats(s);
    };

    auto add_strings_attr = [&xnode](const std::string& name, const std::vector<std::string>& strings) {
        if (strings.empty()) return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::STRINGS);
        for (const std::string& s : strings) xattr->add_strings(s);
    };

    auto add_dtype_attr = [&xnode, add_int_attr](const std::string& name, Dtype v) {
        add_int_attr(name, static_cast<int>(v.ToONNX()));
    };
''')

    lines.append('switch (op_type_) {')
    for node in NODES:
        lines.append('case k%s: ' % (node.op_type) + '{')
        for _, attr in sorted(node.attr_defs.items()):
            lines.append('if (was_%s_set_)' % (attr.c_name))
            lines.append('    %s("%s",' % (attr.add_func(), attr.onnx_name) +
                         ' %s_);' % (attr.c_name))
        lines.append('break;')
        lines.append('}')
    lines.append('}')

    lines.append('for (const onnx::AttributeProto& xattr : unknown_attributes_) {')
    lines.append('*xnode->add_attribute() = xattr;')
    lines.append('}')

    lines.append('}')

    lines.append('void NodeBase::SetDefaultAttributeValues() {')
    lines.append('const float inf = std::numeric_limits<float>::infinity();')
    lines.append('switch (op_type_) {')
    for node in NODES:
        lines.append('case k%s: ' % (node.op_type) + '{')
        for _, attr in sorted(node.attr_defs.items()):
            if attr.value is None:
                continue
            if attr.type == str:
                lines.append('%s_ = "%s";' % (attr.c_name, attr.value))
            elif attr.type == bool:
                lines.append('%s_ = %s;' % (attr.c_name, str(attr.value).lower()))
            else:
                lines.append('%s_ = %s;' % (attr.c_name, attr.value))
        lines.append('break;')
        lines.append('}')
    lines.append('}')
    lines.append('}')

    lines.append('void NodeBase::ValidateNumInputsOutputs('
                 'const std::vector<Value*>& inputs, '
                 'const std::vector<Value*>& outputs) const {')
    lines.append('switch (op_type_) {')
    for node in NODES:
        op = node.op_type

        lines.append('case k%s: ' % (op) + '{')
        for sym, num in [('inputs', node.num_inputs),
                         ('outputs', node.num_outputs)]:
            if isinstance(num, tuple):
                conds = ['%d == %s.size()' % (n, sym) for n in num]
                cond = ' || '.join(conds)
                lines.append('CHECK(%s) << ' % (cond) +
                             '"Unexpected number of %s for %s (" << ' % (sym, op) +
                             '%s.size() << ")";' % (sym))
            elif num is not None:
                lines.append('CHECK_EQ(%d, %s.size()) << ' % (num, sym) +
                             '"Unexpected number of %s for %s";' % (sym, op))

        lines.append('break;')
        lines.append('}')
    lines.append('}')
    lines.append('}')

    lines.append('void NodeBase::ValidateAttributes() const {')
    lines.append('switch (op_type_) {')
    for node in NODES:
        op = node.op_type
        lines.append('case k%s: ' % (op) + '{')
        for key, value in node.attributes.items():
            if isinstance(value, Required):
                lines.append(
                    'CHECK(was_%s_set_) << "%s is mandatory for %s";' % (key, key, op))
        lines.append('break;')
        lines.append('}')
    lines.append('}')
    lines.append('}')

    for attr in ATTRS:
        name = attr.c_name
        arg = attr.c_setter_arg_type()
        lines.append('NodeBase* NodeBase::set_%s(%s %s) ' % (name, arg, name) + '{')
        cond = ' || '.join('op_type_ == k%s' % (t) for t in attr.op_types)
        lines.append('CHECK(%s) << "Invalid attribute `%s\' for " ' % (cond, name) +
                     '<< OpTypeToString(op_type_);')
        if attr.type in [Tensor, Graph]:
            lines.append('%s_.reset(%s);' % (name, name))
        else:
            lines.append('%s_ = %s;' % (name, name))

        lines.append('was_%s_set_ = true;' % (name))
        lines.append('return this;')
        lines.append('}')

    with open('gen_node_base.cc', 'w') as f:
        f.write(r'''// Auto-generated by gen_node.py

#include "gen_node_base.h"

#include <limits>
#include <string>
#include <vector>

#include <onnx/onnx_pb.h>

#include <common/log.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/tensor.h>

namespace oniku {

''')
        f.writelines(codegen_util.format_code(lines))
        f.write(r'''

}  // namespace oniku
''')


gen_gen_node_base_h()
gen_gen_node_base_cc()
