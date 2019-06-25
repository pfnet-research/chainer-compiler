"""Generates boilerplate code for chainer_compiler::Node class.

Nodes in ONNX are very flexible. They allow arbitrary strings as their
operation type (e.g., "Conv") and attribute keys (e.g., "pads"). As we
would limit and expand pre-defined sets of ONNX operations and
attributes, this file will be the definition of what are supported.
"""

import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'common'))
import codegen_util


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


CHAINER_COMPILERX_GLOBAL_ATTRS = attr_sets(chainer_order=-1, chainer_fusion_group=0)

NODES = []


class NodeDef(object):

    def __init__(self, op_type, num_inputs, num_outputs, **kwargs):
        self.op_type = op_type
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.attributes = kwargs
        self.attributes.update(CHAINER_COMPILERX_GLOBAL_ATTRS)
        self.attr_defs = {}  # To be filled after parsed.
        NODES.append(self)


NodeDef('Identity', 1, 1)
NodeDef('Neg', 1, 1)
NodeDef('Reciprocal', 1, 1)
NodeDef('Exp', 1, 1)
NodeDef('Log', 1, 1)
NodeDef('Sqrt', 1, 1)
NodeDef('IsNaN', 1, 1)
NodeDef('IsInf', 1, 1)
NodeDef('Sign', 1, 1)

NodeDef('Sin', 1, 1)
NodeDef('Sinh', 1, 1)
NodeDef('Cos', 1, 1)
NodeDef('Cosh', 1, 1)
NodeDef('Tan', 1, 1)
NodeDef('Tanh', 1, 1)
NodeDef('Asin', 1, 1)
NodeDef('Asinh', 1, 1)
NodeDef('Acos', 1, 1)
NodeDef('Acosh', 1, 1)
NodeDef('Atan', 1, 1)
NodeDef('Atanh', 1, 1)
NodeDef('Erf', 1, 1)

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
NodeDef('And', 2, 1)
NodeDef('Or', 2, 1)
NodeDef('Xor', 2, 1)

NodeDef('Constant', 0, 1, tensor_value=Required(Tensor), chainer_host=False)
NodeDef('ConstantOfShape', 1, 1, tensor_value=Tensor)
# TODO(hamaji): Remove this operator since this op was deprecated.
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
# This Slice supports both Slice-1 and Slice-10.
NodeDef('Slice', (1, 3, 4, 5), 1,
        axes=[int], starts=[int], ends=[int])
# TOOD(hamaji): Remove this as it is deprecated in ONNX.
NodeDef('DynamicSlice', (3, 4, 5), 1)
NodeDef('Gather', 2, 1, axis=0)
NodeDef('Concat', None, 1, axis=Required(int))
NodeDef('Split', 1, None, axis=0, split=[int])
NodeDef('Transpose', 1, 1, perm=[int])
NodeDef('EyeLike', 1, 1, dtype=Dtype, k=0)
NodeDef('DepthToSpace', 1, 1, blocksize=Required(int))
NodeDef('SpaceToDepth', 1, 1, blocksize=Required(int))

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
NodeDef('BatchNormalization', 5, (1, 2, 3, 4, 5, 6),
        epsilon=1e-5, momentum=0.9, spatial=1, chainer_in_recomputing=0)
# Extension: the second output is for backward context.
NodeDef('LRN', 1, (1, 2), alpha=1e-4, beta=0.75, bias=1.0, size=Required(int))
NodeDef('LpNormalization', 1, 1, axis=-1, p=2)

pool_attrs = attr_sets(auto_pad='NOTSET',
                       kernel_shape=Required([int]),
                       pads=[int],
                       storage_order=0,
                       strides=[int])
# Extension: the third output is for backward context.
NodeDef('MaxPool', 1, (1, 2, 3), chainer_cover_all=False, **pool_attrs)
# Extension: the second output is for backward context.
NodeDef('AveragePool', 1, (1, 2), count_include_pad=False, **pool_attrs)
NodeDef('GlobalMaxPool', 1, 1)
NodeDef('GlobalAveragePool', 1, 1)
NodeDef('Pad', 1, 1, mode='constant', pads=[int], value=0.0)
NodeDef('Upsample', 2, 1, mode='nearest')
NodeDef('Resize', 2, 1, mode='nearest')

NodeDef('Softmax', 1, 1, axis=1, chainer_is_onnx_semantics=True)
NodeDef('LogSoftmax', 1, 1, axis=1, chainer_is_onnx_semantics=True)
# Extension: it takes N+1 inputs.
NodeDef('If', None, None, else_branch=Graph, then_branch=Graph)
NodeDef('Loop', None, None, body=Graph, chainer_stack_axis=0)
# TODO(hamaji): Fix Scan to handle the new semantics.
# NodeDef('Scan', None, None, body=Graph, num_scan_inputs=Required(int))
NodeDef('Where', 3, 1)

NodeDef('ImageScaler', 1, 1, scale=1.0, bias_list=[float])
NodeDef('MaxRoiPool', 2, 1, pooled_shape=Required([int]), spatial_scale=1.0)

NodeDef('QuantizeLinear', (2, 3), 1)
NodeDef('DequantizeLinear', (2, 3), 1)
NodeDef('QLinearConv', (8, 9), 1,
        dilations=[int],
        group=1,
        kernel_shape=[int],
        pads=[int],
        strides=[int])
NodeDef('QLinearMatMul', 8, 1)
NodeDef('MatMulInteger', (2, 3, 4), 1)
NodeDef('ConvInteger', (2, 3, 4), 1,
        dilations=[int],
        group=1,
        kernel_shape=[int],
        pads=[int],
        strides=[int])

NodeDef('ChainerLinear', (2, 3), 1, n_batch_axes=1)
NodeDef('ChainerLinearGradWeight', 2, 1)
NodeDef('ChainerReluGrad', 2, 1)
NodeDef('ChainerReduceSumTo', 2, 1)

NodeDef('ChainerROIMaxPool2D', 3, 1,
        output_shape=[int], spatial_scale=Required(float))
NodeDef('ChainerROIAveragePool2D', 3, 1,
        output_shape=[int], spatial_scale=Required(float))
NodeDef('ChainerROIMaxAlign2D', 3, 1,
        output_shape=[int], spatial_scale=Required(float), sampling_ratio=[int])
NodeDef('ChainerROIAverageAlign2D', 3, 1,
        output_shape=[int], spatial_scale=Required(float), sampling_ratio=[int])
NodeDef('ChainerResizeImages', 1, 1, output_shape=[int])

NodeDef('ChainerPadBatchSize', 1, 1, size=Required(int))

# For experimental ops.
NodeDef('ChainerDoSomething', None, None, function_name=Required(str))

NodeDef('ChainerMaxPoolGrad', 2, 1, chainer_cover_all=False, **pool_attrs)
NodeDef('ChainerAveragePoolGrad', 2, 1, count_include_pad=False, **pool_attrs)
NodeDef('ChainerBatchNormalizationGrad', 2, 3)
NodeDef('ChainerConvTransposeWithDynamicOutputShape', 3, 1, **conv_attrs)
NodeDef('ChainerSoftmaxCrossEntropy', 2, 1)
NodeDef('ChainerSelectItem', 2, 1)
NodeDef('ChainerSelectItemGrad', 3, 1)
NodeDef('ChainerLRNGrad', 4, 1,
        alpha=1e-4, beta=0.75, bias=1.0, size=Required(int))
NodeDef('ChainerLSTMGrad', 2, 4)
NodeDef('ChainerConvGradWeight', 3, 1, **conv_attrs)
NodeDef('ChainerGatherGrad', 3, 1, axis=0)
NodeDef('ChainerDynamicSliceGrad', (4, 5, 6), 1)
NodeDef('ChainerFusionGroup', None, None, subgraph=Graph, fusion_type=str)

# Numpy's advanced indexing.
#
# The first input is the tensor to be sliced.
# The number of slices must be as same as the number of slice_specs.
#
# for slice_spec in slice_specs:
#   slice_spec = 0: no input is consumed, i.e., `arr[:]`
#   slice_spec = 1: the next input must be a Tensor, i.e., `arr[2]`
#   slice_spec = 2: the next input must be two Tensors, i.e., `arr[2:3]`
#   slice_spec = 3: the next input must be three Tensor, i.e., `arr[2:10:2]`
#   slice_spec = 4: the next input must be a Sequence, i.e., `arr[[1,2,3]]`
#
# For example, T[:, Seq, I1, I2:I3, I4:I5:I6] must be encoded as
#
# ChainerGetItem(T, Seq, I1, I2, I3, I4, I5, I6, slice_specs=[0, 4, 1, 2, 3])
NodeDef('ChainerGetItem', None, 1, slice_specs=[int])
# One more inputs for the shape info.
NodeDef('ChainerGetItemGrad', None, 1, slice_specs=[int])

NodeDef('ChainerPrint', None, 0)

# Put a null value.
NodeDef('ChainerNullConstant', 0, 1)

# Creates a constant sequence: () -> ([T])
NodeDef('ChainerSequenceConstants', 0, 1, tensor_values=[Tensor])

# Creates a new sequence: (T...) -> ([T])
NodeDef('ChainerSequenceCreate', None, 1)

# Appends an element to a sequence: ([T], T) -> ([T])
NodeDef('ChainerSequenceAppend', 2, 1)

# Pops an element from a sequence: ([T]) -> ([T], T)
NodeDef('ChainerSequencePop', 1, 2)

# Looks up an element in a sequence: ([T], I) -> (T)
NodeDef('ChainerSequenceLookup', 2, 1)

# Equivalent to Python's __getitem__ for a slice: ([T], I, I, I) -> ([T])
NodeDef('ChainerSequenceGetSlice', (1, 2, 3, 4), 1)

# Stacks elements in a sequence: ([T]) -> (T)
NodeDef('ChainerSequenceStack', 1, 1, axis=0)

# Concatenates elements in a sequence: ([T]) -> (T)
# The second output is for backward context.
NodeDef('ChainerSequenceConcat', 1, (1, 2), axis=0)

# Splits a tensor to a sequence (like F.split_axis): (T, I) -> ([T])
NodeDef('ChainerSequenceSplitAxis', 2, 1, axis=0)

# Pads elements in a sequence: ([T]) -> (T)
NodeDef('ChainerSequencePad', 1, 1, length=0, value=0.0)

# Splits a tensor to a sequence (like F.separate): (T) -> ([T])
NodeDef('ChainerSequenceSeparate', 1, 1, axis=0)

# Strips paddings in a tensor and returns a sequence: (T, [I]) -> ([T])
# Note the result of SequenceLengths can be used as the second argument.
NodeDef('ChainerSequenceUnpad', 2, 1)

# Returns the number of elements in a sequence: ([T]) -> (I)
NodeDef('ChainerSequenceSize', 1, 1)

# Returns lengths of elements in a sequence: ([T]) -> ([I])
NodeDef('ChainerSequenceLengths', 1, 1)

# Equivalent to Python's range.
NodeDef('ChainerSequenceRange', (1, 2, 3), 1)

# The gradients of sequence related ops.
NodeDef('ChainerSequenceLookupGrad', 3, 1)
NodeDef('ChainerSequenceGetSliceGrad', (2, 3, 4, 5), 1)

# Equivalent to Python's __len__.
# For tensors: Gather(Shape(input0), 0)
# For sequences: ChainerSequenceSize(input0)
NodeDef('ChainerGenericLen', 1, 1)

# Equivalent to Python's __getitem__ for a scalar index.
# For tensors: Gather(input0, input1)
# For sequences: ChainerSequenceLookup(input0, input1)
# TODO(hamaji): Deprecate this op.
NodeDef('ChainerGenericGetItem', 2, 1)

# Equivalent to Python's __getitem__ for a slice.
# For tensors: DynamicSlice(input0, [input1], [input2]) -> tensor
# For sequences: input0[input1:input2:input3] in Python -> sequence
# TODO(hamaji): Deprecate this op.
NodeDef('ChainerGenericGetSlice', (1, 2, 3, 4), 1)

# Equivalent to Python's __add__.
# For tensors: Add(input0, input1) -> tensor
# For sequences: input0 + input1 in Python -> sequence
# TODO(hamaji): Deprecate this op.
NodeDef('ChainerGenericAdd', 2, 1)

# Similar to Python's `is` keyword.
# This returns true only when both inputs are bool scalars and have
# the same value.
NodeDef('ChainerGenericIs', 2, 1)

# Accumulates two gradient values.
# For tensors: Add(input0, input1) -> tensor
# For sequence: Add(i0, i1) for each element in sequences.
NodeDef('ChainerGenericAccumulateGrad', 2, 1)


class AttrDef(object):
    def __init__(self, name, value):
        self.name = name
        self.onnx_name = self.name
        if self.onnx_name == 'tensor_value':
            self.onnx_name = 'value'
        elif self.onnx_name == 'bias_list':
            self.onnx_name = 'bias'
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
        if self.type == [Tensor]:
            return 'std::vector<std::unique_ptr<Tensor>>&&'
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
    public_lines.append('static OpType StringToOpType(const std::string& str);')
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

#include <compiler/dtype.h>
#include <compiler/onnx.h>

namespace chainer_compiler {

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

}  // namespace chainer_compiler
''')


def gen_gen_node_base_cc():
    lines = []

    lines.append('NodeBase::OpType NodeBase::StringToOpType'
                 '(const std::string& str) {')
    conds = []
    bodies = []
    for i, node in enumerate(NODES):
        conds.append('str == "%s"' % node.op_type)
        bodies.append(['return k%s;' % node.op_type])
    bodies.append(['CHECK(false) << "Unsupported op_type: " << str;'])
    lines.extend(codegen_util.cond(conds, bodies))
    lines.append('}')

    lines.append('NodeBase::NodeBase(OpType op_type) : op_type_(op_type) {}')
    lines.append('NodeBase::NodeBase(const onnx::NodeProto& xnode, '
                 'const std::vector<Value*>& inputs, '
                 'const std::vector<Value*>& outputs) {')
    lines.append('op_type_ = StringToOpType(xnode.op_type());')

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
            blines.append(
                'if (!g_permissive) '
                'CHECK_EQ(xattr.type(), %s) << xnode.DebugString();' %
                attr.onnx_type())
            if attr.type == int:
                blines.append('set_%s(xattr.i());' % (attr.c_name))
            elif attr.type == bool:
                blines.append('set_%s(xattr.i() != 0);' % (attr.c_name))
            elif attr.type == float:
                blines.append('set_%s(xattr.f());' % (attr.c_name))
            elif attr.type == str:
                blines.append('set_%s(xattr.s());' % (attr.c_name))
            elif attr.type == [Tensor]:
                blines.append('for (const auto& t : xattr.tensors()) '
                              '%s_.emplace_back(new Tensor(t));' % attr.c_name)
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

    auto add_tensors_attr = [&xnode](const std::string& name, const std::vector<std::unique_ptr<Tensor>>& vec) {
        if (vec.empty()) return;
        onnx::AttributeProto* xattr = xnode->add_attribute();
        xattr->set_name(name);
        xattr->set_type(onnx::AttributeProto::TENSORS);
        for (const std::unique_ptr<Tensor>& t : vec) t->ToONNX(xattr->add_tensors());
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
        elif attr.type == [Tensor]:
            lines.append('%s_ = std::move(%s);' % (name, name))
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

#include <common/log.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/onnx.h>
#include <compiler/tensor.h>

namespace chainer_compiler {

''')
        f.writelines(codegen_util.format_code(lines))
        f.write(r'''

}  // namespace chainer_compiler
''')


gen_gen_node_base_h()
gen_gen_node_base_cc()
