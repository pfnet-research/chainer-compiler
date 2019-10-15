#include "compiler/chxvm/simple_node_emitter.h"

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/chxvm/chxvm_value.h>
#include <compiler/chxvm/value_id_manager.h>
#include <compiler/flops.h>
#include <compiler/gen_chxvm_codegen.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>
#include <compiler/value.h>
#include <runtime/chxvm.pb.h>

namespace chainer_compiler {
namespace chxvm {

using chainer_compiler::runtime::ChxVMProgramProto;

namespace {

// TODO(hamaji): Move this to the middle end, not codegen.
std::vector<int64_t> ComplementStrideOrPad(const std::vector<int64_t>& orig, const Value* input, int64_t default_value) {
    const Type& type = input->type();
    // Fill strides or pads for statically known input shape.
    if (!orig.empty() || !type.HasKnownShape()) {
        return orig;
    }
    std::vector<int64_t> filled;
    CHECK_LT(2, type.ndim()) << type.DebugString();
    for (int i = 0; i < type.ndim() - 2; ++i) {
        filled.push_back(default_value);
    }
    return filled;
}

void FillOpInfo(const Node& node, const std::string& debug_info, ChxVMProgramProto* prog) {
    runtime::ChxVMInstructionProto* inst = prog->mutable_instructions(prog->instructions_size() - 1);
    inst->set_debug_info(debug_info);
    inst->set_id(node.chainer_order());
    inst->set_flops(CalculateFlops(node));
}

#define EMIT(op, ...)                            \
    do {                                         \
        Add##op##Op(prog, __VA_ARGS__);          \
        FillOpInfo(node, node.ToString(), prog); \
    } while (0);

}  // namespace

void EmitSimpleNode(const Node& node, const ValueIdManager& id_manager, ChxVMProgramProto* prog) {
    auto in = [&node, &id_manager](int i) {
        CHECK_LT(i, node.inputs().size()) << i << "th input of " << node.op_type() << " is mandatory: " << node.DebugString();
        Value* input = node.input(i);
        CHECK(!input->IsNull()) << i << "th input of " << node.op_type() << " is mandatory: " << node.DebugString();
        return id_manager.GetValueId(input);
    };

    // Optional input.
    auto oin = [in, &node](int i) {
        if (i >= static_cast<int>(node.inputs().size())) return -1;
        if (node.input(i)->IsNull()) return -1;
        return in(i);
    };

    auto out = [&node, &id_manager](int i) { return ChxVMValue::GetOutputValue(node, i, id_manager); };

    // Optional output.
    auto oout = [out, &node](int i) {
        if (i >= static_cast<int>(node.outputs().size())) return ChxVMValue(-1);
        if (node.output(i)->IsNull()) return ChxVMValue(-1);
        return out(i);
    };

    auto pads = [&node]() {
        std::vector<int64_t> pads = node.pads();
        // Both Chainer and ChainerX expect paddings for beginning
        // and end are the same.
        CHECK_EQ(pads.size() % 2, 0);
        for (size_t i = 0; i < pads.size() / 2; ++i) {
            CHECK_EQ(pads[i], pads[i + pads.size() / 2]);
        }
        pads.resize(pads.size() / 2);
        return ComplementStrideOrPad(pads, node.input(0), 0);
    };

    auto strides = [&node]() {
        std::vector<int64_t> strides = node.strides();
        return ComplementStrideOrPad(strides, node.input(0), 1);
    };

    auto auto_pad = [&node]() {
        if (node.auto_pad() == "NOTSET") {
            return std::string();
        }
        // Support auto_pad only for MNIST
        CHECK_EQ(node.auto_pad(), "SAME_UPPER");
        return node.auto_pad();
    };

    auto direction = [&node]() {
        const std::string& dir = node.direction();
        if (dir == "" || dir == "forward")
            return 0;
        else if (dir == "reverse")
            return 1;
        else if (dir == "bidirectional")
            return 2;
        else
            CHECK(false) << "Unknown direction: " << dir << ": " << node.DebugString();
    };

#define EMIT_SIMPLE_UNARY_OP(name, sym)           \
    do {                                          \
        if (node.op_type() == name) {             \
            CHECK_EQ(1UL, node.inputs().size());  \
            CHECK_EQ(1UL, node.outputs().size()); \
            EMIT(sym, out(0), in(0));             \
            return;                               \
        }                                         \
    } while (0)

#define EMIT_SIMPLE_BINARY_OP(name, sym)          \
    do {                                          \
        if (node.op_type() == name) {             \
            CHECK_EQ(2UL, node.inputs().size());  \
            CHECK_EQ(1UL, node.outputs().size()); \
            EMIT(sym, out(0), in(0), in(1));      \
            return;                               \
        }                                         \
    } while (0)

    EMIT_SIMPLE_UNARY_OP(Node::kNeg, Neg);
    EMIT_SIMPLE_UNARY_OP(Node::kReciprocal, Reciprocal);
    EMIT_SIMPLE_UNARY_OP(Node::kExp, Exp);
    EMIT_SIMPLE_UNARY_OP(Node::kLog, Log);
    EMIT_SIMPLE_UNARY_OP(Node::kSqrt, Sqrt);
    EMIT_SIMPLE_UNARY_OP(Node::kSin, Sin);
    EMIT_SIMPLE_UNARY_OP(Node::kSinh, Sinh);
    EMIT_SIMPLE_UNARY_OP(Node::kCos, Cos);
    EMIT_SIMPLE_UNARY_OP(Node::kCosh, Cosh);
    EMIT_SIMPLE_UNARY_OP(Node::kTan, Tan);
    EMIT_SIMPLE_UNARY_OP(Node::kTanh, Tanh);
    EMIT_SIMPLE_UNARY_OP(Node::kAsin, Arcsin);
    EMIT_SIMPLE_UNARY_OP(Node::kAsinh, Arcsinh);
    EMIT_SIMPLE_UNARY_OP(Node::kAcos, Arccos);
    EMIT_SIMPLE_UNARY_OP(Node::kAcosh, Arccosh);
    EMIT_SIMPLE_UNARY_OP(Node::kAtan, Arctan);
    EMIT_SIMPLE_UNARY_OP(Node::kAtanh, Arctanh);
    EMIT_SIMPLE_UNARY_OP(Node::kErf, Erf);
    EMIT_SIMPLE_UNARY_OP(Node::kAbs, Abs);
    EMIT_SIMPLE_UNARY_OP(Node::kRelu, Relu);
    EMIT_SIMPLE_UNARY_OP(Node::kFloor, Floor);
    EMIT_SIMPLE_UNARY_OP(Node::kCeil, Ceil);
    EMIT_SIMPLE_UNARY_OP(Node::kSigmoid, Sigmoid);
    EMIT_SIMPLE_UNARY_OP(Node::kNot, Not);
    EMIT_SIMPLE_UNARY_OP(Node::kIdentity, Identity);
    EMIT_SIMPLE_UNARY_OP(Node::kIsNaN, IsNaN);
    EMIT_SIMPLE_UNARY_OP(Node::kSign, Sign);
    EMIT_SIMPLE_UNARY_OP(Node::kRound, Round);
    EMIT_SIMPLE_UNARY_OP(Node::kSoftplus, Softplus);

    EMIT_SIMPLE_BINARY_OP(Node::kAdd, Add);
    EMIT_SIMPLE_BINARY_OP(Node::kSub, Sub);
    EMIT_SIMPLE_BINARY_OP(Node::kMul, Mul);
    EMIT_SIMPLE_BINARY_OP(Node::kDiv, Div);
    EMIT_SIMPLE_BINARY_OP(Node::kPow, Pow);
    EMIT_SIMPLE_BINARY_OP(Node::kEqual, Equal);
    EMIT_SIMPLE_BINARY_OP(Node::kGreater, Greater);
    EMIT_SIMPLE_BINARY_OP(Node::kChainerGenericIs, GenericIs);
    EMIT_SIMPLE_BINARY_OP(Node::kAnd, And);
    EMIT_SIMPLE_BINARY_OP(Node::kOr, Or);
    EMIT_SIMPLE_BINARY_OP(Node::kXor, Xor);

    EMIT_SIMPLE_BINARY_OP(Node::kChainerReluGrad, ReluGrad);
    EMIT_SIMPLE_BINARY_OP(Node::kChainerSelectItem, SelectItem);

    if (node.op_type() == Node::kMod) {
        if (node.fmod()) {
            EMIT(Fmod, out(0), in(0), in(1));
        } else {
            EMIT(Mod, out(0), in(0), in(1));
        }
    } else if (node.op_type() == Node::kDropout) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_LE(1UL, node.outputs().size());
        CHECK_GE(2UL, node.outputs().size());
        if (node.outputs().size() >= 2UL) {
            WARN_ONCE("The second output of Dropout is not handled yet");
        }
        EMIT(Dropout, out(0), oout(1), in(0), node.ratio());
    } else if (node.op_type() == Node::kSelu) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_LE(1UL, node.outputs().size());
        EMIT(Selu, out(0), in(0), node.alpha(), node.gamma());
    } else if (node.op_type() == Node::kIsInf) {
        EMIT(IsInf, out(0), in(0), node.detect_negative(), node.detect_positive());
    } else if (node.op_type() == Node::kLeakyRelu) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_LE(1UL, node.outputs().size());
        EMIT(LeakyRelu, out(0), in(0), node.alpha());
    } else if (node.op_type() == Node::kElu) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_LE(1UL, node.outputs().size());
        EMIT(Elu, out(0), in(0), node.alpha());
    } else if (node.op_type() == Node::kChainerLinear) {
        EMIT(Linear, out(0), in(0), in(1), oin(2), node.n_batch_axes());
    } else if (node.op_type() == Node::kChainerLinearGradWeight) {
        EMIT(LinearGradWeight, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kConv) {
        CHECK_LE(2UL, node.inputs().size());
        CHECK_GE(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        // TODO(ChainerX): Support dilation.
        for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
        EMIT(Conv, out(0), in(0), in(1), oin(2), strides(), pads(), node.group(), auto_pad());
    } else if (node.op_type() == Node::kConvTranspose) {
        CHECK_LE(2UL, node.inputs().size());
        CHECK_GE(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        // TODO(ChainerX): Support dilation.
        for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
        // TODO(hamaji): Handle output_padding and output_shape.
        std::vector<int64_t> output_shape(node.output_shape());
        EMIT(ConvTranspose, out(0), in(0), in(1), oin(2), strides(), pads(), node.group(), output_shape);
    } else if (node.op_type() == Node::kChainerConvTransposeWithDynamicOutputShape) {
        CHECK_EQ(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(ConvTransposeWithDynamicShape, out(0), in(0), in(1), in(2), strides(), pads(), node.group());
    } else if (node.op_type() == Node::kChainerConvGradWeight) {
        CHECK_EQ(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        // TODO(ChainerX): Support dilation.
        for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
        EMIT(ConvGradWeight, out(0), in(0), in(1), in(2), strides(), pads(), node.group());
    } else if (node.op_type() == Node::kRNN) {
        CHECK(node.activations().empty()) << "activations not supporte yet";
        CHECK(node.activation_alpha().empty()) << "activation_alpha not supporte yet";
        CHECK(node.activation_beta().empty()) << "activation_beta not supporte yet";
        EMIT(RNN, oout(0), oout(1), in(0), in(1), in(2), oin(3), oin(4), oin(5), node.hidden_size(), direction());
    } else if (node.op_type() == Node::kGRU) {
        CHECK(node.activations().empty()) << "activations not supporte yet";
        CHECK(node.activation_alpha().empty()) << "activation_alpha not supporte yet";
        CHECK(node.activation_beta().empty()) << "activation_beta not supporte yet";
        EMIT(GRU,
             oout(0),
             oout(1),
             in(0),
             in(1),
             in(2),
             oin(3),
             oin(4),
             oin(5),
             node.hidden_size(),
             node.linear_before_reset(),
             direction());
    } else if (node.op_type() == Node::kLSTM) {
        CHECK(node.activations().empty()) << "activations not supporte yet";
        CHECK(node.activation_alpha().empty()) << "activation_alpha not supporte yet";
        CHECK(node.activation_beta().empty()) << "activation_beta not supporte yet";
        EMIT(LSTM,
             oout(0),
             oout(1),
             oout(2),
             oout(3),
             in(0),
             in(1),
             in(2),
             oin(3),
             oin(4),
             oin(5),
             oin(6),
             oin(7),
             node.hidden_size(),
             direction());
    } else if (node.op_type() == Node::kChainerLSTMGrad) {
        EMIT(LSTMGrad, out(0), out(1), out(2), out(3), in(0), in(1));
    } else if (node.op_type() == Node::kShape) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Shape, out(0), in(0));
    } else if (node.op_type() == Node::kSize) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Size, out(0), in(0));
    } else if (node.op_type() == Node::kReshape) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Reshape, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kExpand) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Expand, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kSqueeze) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Squeeze, out(0), in(0), node.axes());
    } else if (node.op_type() == Node::kUnsqueeze) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        std::vector<int64_t> axes = node.axes();
        if (!axes.empty()) {
            std::sort(axes.begin(), axes.end());
            if (axes.front() < 0 && axes.back() > 0) {
                CHECK(false) << "Unsqueeze with negative/positive axes mixed is not supported: " << JoinString(node.axes());
            }
        }
        EMIT(Unsqueeze, out(0), in(0), axes);
    } else if (node.op_type() == Node::kMatMul) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(MatMul, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kGemm) {
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Gemm, out(0), in(0), in(1), oin(2), node.alpha(), node.beta(), node.trans_a(), node.trans_b());
    } else if (node.op_type() == Node::kLRN) {
        EMIT(LRN, out(0), oout(1), in(0), node.alpha(), node.beta(), node.bias(), node.size());
    } else if (node.op_type() == Node::kChainerLRNGrad) {
        EMIT(LRNGrad, out(0), in(0), in(1), in(2), in(3), node.alpha(), node.beta(), node.bias(), node.size());
    } else if (node.op_type() == Node::kUpsample) {
        CHECK_EQ("nearest", node.mode()) << "Only nearest upsampling is supported";
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Resize, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kResize) {
        CHECK_EQ("nearest", node.mode()) << "Only nearest upsampling is supported";
        // TODO(take-cheeze): Handle Resize-11
        CHECK_EQ(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Resize, out(0), in(0), in(2));
    } else if (node.op_type() == Node::kChainerResizeGrad) {
        EMIT(ResizeGrad, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kPad) {
        CHECK_EQ(1UL, node.outputs().size());
        if (node.inputs().size() == 1) {
            CHECK_EQ("constant", node.mode()) << "Only constant padding is supported";
            EMIT(Pad, out(0), in(0), node.pads(), node.value());
        } else {
            CHECK_EQ(0, node.pads().size());
            CHECK_EQ(0.0, node.value());
            EMIT(DynamicPad, out(0), in(0), in(1), oin(2));
        }
    } else if (node.op_type() == Node::kMaxPool) {
        CHECK_EQ(1UL, node.inputs().size());
        if (node.outputs().size() != 1) {
            CHECK_EQ(3UL, node.outputs().size());
            CHECK(node.output(1)->IsNull());
        }
        EMIT(MaxPool, out(0), oout(2), in(0), node.kernel_shape(), strides(), pads(), node.ceil_mode(), auto_pad());
    } else if (node.op_type() == Node::kChainerMaxPoolGrad) {
        CHECK_EQ("NOTSET", node.auto_pad()) << "auto_pad is not supported for MaxPool";
        EMIT(MaxPoolGrad, out(0), in(0), in(1), node.kernel_shape(), node.ceil_mode());
    } else if (node.op_type() == Node::kChainerROIMaxPool2D) {
        EMIT(ROIMaxPool2D, out(0), in(0), in(1), in(2), node.output_shape(), node.spatial_scale());
    } else if (node.op_type() == Node::kChainerROIAveragePool2D) {
        EMIT(ROIAveragePool2D, out(0), in(0), in(1), in(2), node.output_shape(), node.spatial_scale());
    } else if (node.op_type() == Node::kChainerROIMaxAlign2D) {
        EMIT(ROIMaxAlign2D, out(0), in(0), in(1), in(2), node.output_shape(), node.spatial_scale(), node.sampling_ratio_list());
    } else if (node.op_type() == Node::kChainerROIAverageAlign2D) {
        EMIT(ROIAverageAlign2D, out(0), in(0), in(1), in(2), node.output_shape(), node.spatial_scale(), node.sampling_ratio_list());
    } else if (node.op_type() == Node::kRoiAlign) {
        std::vector<int64_t> sampling_ratio = {node.sampling_ratio(), node.sampling_ratio()};
        std::vector<int64_t> output_shape = {node.output_height(), node.output_width()};
        if (node.mode() == "avg") {
            EMIT(ROIAverageAlign2D, out(0), in(0), in(1), in(2), output_shape, node.spatial_scale(), sampling_ratio);
        } else if (node.mode() == "max") {
            EMIT(ROIMaxAlign2D, out(0), in(0), in(1), in(2), output_shape, node.spatial_scale(), sampling_ratio);
        } else {
            CHECK(false) << "Unknown RoiAlign mode: " << node.mode();
        }
    } else if (node.op_type() == Node::kChainerResizeImages) {
        EMIT(ResizeImages, out(0), in(0), node.output_shape());
    } else if (node.op_type() == Node::kAveragePool) {
        CHECK_EQ("NOTSET", node.auto_pad()) << "auto_pad is not supported for AveragePool";
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(0, node.ceil_mode()) << "ceil_mode for AveragePool is not supported yet";
        EMIT(AveragePool, out(0), oout(1), in(0), node.kernel_shape(), strides(), pads(), node.count_include_pad());
    } else if (node.op_type() == Node::kChainerAveragePoolGrad) {
        CHECK_EQ("NOTSET", node.auto_pad()) << "auto_pad is not supported for AveragePool";
        CHECK_EQ(0, node.ceil_mode()) << "ceil_mode for AveragePool is not supported yet";
        EMIT(AveragePoolGrad, out(0), in(0), in(1), node.kernel_shape(), node.count_include_pad());
    } else if (node.op_type() == Node::kChainerPadBatchSize) {
        EMIT(PadBatchSize, out(0), in(0), node.size());
    } else if (node.op_type() == Node::kSoftmax) {
        EMIT(Softmax, out(0), in(0), node.axis(), node.chainer_is_onnx_semantics());
    } else if (node.op_type() == Node::kLogSoftmax) {
        EMIT(LogSoftmax, out(0), in(0), node.axis(), node.chainer_is_onnx_semantics());
    } else if (node.op_type() == Node::kArgMax) {
        EMIT(ArgMax, out(0), in(0), node.axis(), node.keepdims());
    } else if (node.op_type() == Node::kHardmax) {
        EMIT(Hardmax, out(0), in(0), node.axis());
    } else if (node.op_type() == Node::kReduceMax) {
        EMIT(ReduceMax, out(0), in(0), node.axes(), node.keepdims());
    } else if (node.op_type() == Node::kReduceMin) {
        EMIT(ReduceMin, out(0), in(0), node.axes(), node.keepdims());
    } else if (node.op_type() == Node::kReduceSum) {
        EMIT(ReduceSum, out(0), in(0), node.axes(), node.keepdims());
    } else if (node.op_type() == Node::kReduceSumSquare) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(ReduceSumSquare, out(0), in(0), node.axes(), node.keepdims());
    } else if (node.op_type() == Node::kChainerReduceSumTo) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(ReduceSumTo, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kReduceMean) {
        EMIT(ReduceMean, out(0), in(0), node.axes(), node.keepdims());
    } else if (node.op_type() == Node::kReduceProd) {
        EMIT(ReduceProd, out(0), in(0), node.axes(), node.keepdims());
    } else if (node.op_type() == Node::kCast) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Cast, out(0), in(0), node.to());
    } else if (node.op_type() == Node::kChainerDynamicCast) {
        EMIT(DynamicCast, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kChainerDtype) {
        EMIT(Dtype, out(0), in(0));
    } else if (node.op_type() == Node::kOneHot) {
        EMIT(OneHot, out(0), in(0), in(1), in(2), node.axis());
    } else if (node.op_type() == Node::kConstantFill) {
        if (node.input_as_shape()) {
            CHECK_EQ(1UL, node.inputs().size());
        } else {
            CHECK_EQ(0UL, node.inputs().size());
        }
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(ConstantFill, out(0), oin(0), node.dtype(), node.extra_shape(), node.shape(), node.value());
    } else if (node.op_type() == Node::kEyeLike) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(EyeLike, out(0), in(0), node.dtype(), node.k());
    } else if (node.op_type() == Node::kSlice) {
        CHECK_EQ(1UL, node.outputs().size());
        if (node.inputs().size() > 1) {
            // Slice-10
            CHECK_EQ(0UL, node.axes().size());
            CHECK_EQ(0UL, node.starts().size());
            CHECK_EQ(0UL, node.ends().size());
            EMIT(DynamicSlice, out(0), in(0), in(1), in(2), oin(3), oin(4));
        } else {
            // Slice-1
            CHECK_EQ(1UL, node.inputs().size());
            CHECK_NE(0UL, node.starts().size());
            CHECK_NE(0UL, node.ends().size());
            CHECK_EQ(node.starts().size(), node.ends().size());
            std::vector<int64_t> axes(node.axes());
            if (axes.empty()) {
                for (size_t i = 0; i < node.starts().size(); ++i) axes.push_back(i);
            } else {
                CHECK_EQ(node.starts().size(), axes.size());
            }
            EMIT(Slice, out(0), in(0), axes, node.starts(), node.ends());
        }
    } else if (node.op_type() == Node::kDynamicSlice) {
        EMIT(DynamicSlice, out(0), in(0), in(1), in(2), oin(3), oin(4));
    } else if (node.op_type() == Node::kChainerGetItem) {
        std::vector<int> ins;
        for (size_t i = 1; i < node.inputs().size(); ++i) ins.push_back(in(i));
        EMIT(GetItem, out(0), in(0), ins, node.slice_specs());
    } else if (node.op_type() == Node::kChainerSetItem) {
        std::vector<int> ins;
        size_t last_index = node.inputs().size() - 1;
        for (size_t i = 1; i < last_index; ++i) ins.push_back(in(i));
        EMIT(SetItem, out(0), in(0), ins, in(last_index), node.slice_specs());
    } else if (node.op_type() == Node::kChainerGetItemGrad) {
        std::vector<int> ins;
        for (size_t i = 2; i < node.inputs().size(); ++i) ins.push_back(in(i));
        EMIT(GetItemGrad, out(0), in(0), in(1), ins, node.slice_specs());
    } else if (node.op_type() == Node::kGather) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Gather, out(0), in(0), in(1), node.axis());
    } else if (node.op_type() == Node::kGatherElements) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(GatherElements, out(0), in(0), in(1), node.axis());
    } else if (node.op_type() == Node::kGatherND) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(GatherND, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kScatter || node.op_type() == Node::kScatterElements) {
        CHECK_EQ(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Scatter, out(0), in(0), in(1), in(2), node.axis());
    } else if (node.op_type() == Node::kScatterND) {
        CHECK_EQ(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(ScatterND, out(0), in(0), in(1), in(2));
    } else if (node.op_type() == Node::kConcat) {
        CHECK_EQ(1UL, node.outputs().size());
        std::vector<int> ins;
        for (size_t i = 0; i < node.inputs().size(); ++i) ins.push_back(in(i));
        EMIT(Concat, out(0), ins, node.axis());
    } else if (node.op_type() == Node::kChainerConcatGrad) {
        std::vector<int> shapes;
        for (size_t i = 1; i < node.inputs().size(); ++i) shapes.push_back(in(i));
        std::vector<ChxVMValue> outs;
        for (size_t i = 0; i < node.outputs().size(); ++i) outs.push_back(out(i));
        EMIT(ConcatGrad, outs, in(0), shapes, node.axis());
    } else if (node.op_type() == Node::kSplit) {
        CHECK_EQ(1UL, node.inputs().size());
        std::vector<ChxVMValue> outs;
        for (size_t i = 0; i < node.outputs().size(); ++i) outs.push_back(out(i));
        EMIT(Split, outs, in(0), node.axis(), node.split());
    } else if (node.op_type() == Node::kClip) {
        CHECK_EQ(1UL, node.outputs().size());
        if (node.inputs().size() == 1) {
            EMIT(StaticClip, out(0), in(0), node.max(), node.min());
        } else {
            EMIT(Clip, out(0), in(0), oin(1), oin(2));
        }
    } else if (node.op_type() == Node::kMax) {
        CHECK_EQ(1UL, node.outputs().size());
        std::vector<int> ins;
        for (size_t i = 0; i < node.inputs().size(); ++i) ins.push_back(in(i));
        EMIT(Max, out(0), ins);
    } else if (node.op_type() == Node::kMin) {
        CHECK_EQ(1UL, node.outputs().size());
        std::vector<int> ins;
        for (size_t i = 0; i < node.inputs().size(); ++i) ins.push_back(in(i));
        EMIT(Min, out(0), ins);
    } else if (node.op_type() == Node::kTranspose) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Transpose, out(0), in(0), node.perm());
    } else if (node.op_type() == Node::kDepthToSpace) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        bool is_crd = false;
        if (node.mode() != "DCR") {
            CHECK_EQ(node.mode(), "CRD") << "Unsupported DepthToSpace mode";
            is_crd = true;
        }
        EMIT(DepthToSpace, out(0), in(0), node.blocksize(), is_crd);
    } else if (node.op_type() == Node::kSpaceToDepth) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(SpaceToDepth, out(0), in(0), node.blocksize());
    } else if (node.op_type() == Node::kChainerBatchNormalizationGrad) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(3UL, node.outputs().size());
        EMIT(BatchNormalizationGrad, out(0), out(1), out(2), in(0), in(1));
    } else if (node.op_type() == Node::kChainerSelectItemGrad) {
        EMIT(SelectItemGrad, out(0), in(0), in(1), in(2));
    } else if (node.op_type() == Node::kChainerGatherGrad) {
        EMIT(GatherGrad, out(0), in(0), in(1), in(2), node.axis());
    } else if (node.op_type() == Node::kChainerDynamicSliceGrad) {
        EMIT(DynamicSliceGrad, out(0), in(0), in(1), in(2), in(3), oin(4), oin(5));
    } else if (node.op_type() == Node::kChainerDoSomething) {
        std::vector<int> ins;
        std::vector<ChxVMValue> outs;
        for (size_t i = 0; i < node.inputs().size(); ++i) ins.push_back(in(i));
        for (size_t i = 0; i < node.outputs().size(); ++i) outs.push_back(out(i));
        EMIT(DoSomething, outs, ins, node.function_name());
    } else if (node.op_type() == Node::kChainerPrint) {
        std::vector<int> ins;
        for (size_t i = 0; i < node.inputs().size(); ++i) ins.push_back(in(i));
        EMIT(Print, ins);
    } else if (node.op_type() == Node::kSequenceConstruct) {
        std::vector<int> ins;
        for (size_t i = 0; i < node.inputs().size(); ++i) ins.push_back(in(i));
        EMIT(SequenceCreate, out(0), ins);
    } else if (node.op_type() == Node::kSequenceLength) {
        EMIT(SequenceSize, out(0), in(0));
    } else if (node.op_type() == Node::kChainerSequenceLengths) {
        EMIT(SequenceLengths, out(0), in(0));
    } else if (node.op_type() == Node::kSequenceInsert) {
        ChxVMValue o(out(0));
        if (node.inputs().size() == 3) {
            EMIT(SequenceInsert, o, in(0), in(1), in(2));
        } else if (node.input(0)->users().size() == 1) {
            // Avoid O(N^2) copies for the simple case.
            EMIT(SequenceMove, o, in(0));
            EMIT(SequenceAppend, o.id(), in(1));
        } else {
            EMIT(SequenceCopy, o, in(0));
            EMIT(SequenceAppend, o.id(), in(1));
        }
    } else if (node.op_type() == Node::kChainerSequenceExtend) {
        EMIT(SequenceExtend, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kSequenceErase || node.op_type() == Node::kChainerSequencePop) {
        if (node.inputs().size() == 2) {
            EMIT(SequenceErase, out(0), in(0), in(1));
        } else {
            ChxVMValue o0(out(0));
            if (node.input(0)->users().size() == 1) {
                // Avoid O(N^2) copies for the simple case.
                EMIT(SequenceMove, o0, in(0));
                EMIT(SequencePop, out(1), o0.id());
            } else {
                EMIT(SequenceCopy, o0, in(0));
                EMIT(SequencePop, out(1), o0.id());
            }
        }
    } else if (node.op_type() == Node::kSequenceAt) {
        EMIT(SequenceLookup, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kChainerSequenceUpdate) {
        EMIT(SequenceUpdate, out(0), in(0), in(1), in(2));
    } else if (node.op_type() == Node::kChainerSequenceGetSlice) {
        EMIT(SequenceGetSlice, out(0), in(0), oin(1), oin(2), oin(3));
    } else if (node.op_type() == Node::kChainerSequenceAtGrad) {
        EMIT(SequenceLookupGrad, out(0), in(0), in(1), in(2));
    } else if (node.op_type() == Node::kChainerSequenceGetSliceGrad) {
        EMIT(SequenceGetSliceGrad, out(0), in(0), in(1), oin(2), oin(3), oin(4));
    } else if (node.op_type() == Node::kConcatFromSequence) {
        if (node.new_axis()) {
            EMIT(SequenceStack, out(0), in(0), node.axis());
        } else {
            EMIT(SequenceConcat, out(0), oout(1), in(0), node.axis());
        }
    } else if (node.op_type() == Node::kChainerSequenceSplitAxis) {
        EMIT(SequenceSplitAxis, out(0), in(0), in(1), node.axis());
    } else if (node.op_type() == Node::kChainerSequenceSeparate) {
        EMIT(SequenceSeparate, out(0), in(0), node.axis());
    } else if (node.op_type() == Node::kSplitToSequence) {
        if (node.keepdims()) {
            EMIT(SequenceSplitAxis, out(0), in(0), oin(1), node.axis());
        } else {
            EMIT(SequenceSeparate, out(0), in(0), node.axis());
        }
    } else if (node.op_type() == Node::kChainerSequenceUnpad) {
        EMIT(SequenceUnpad, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kChainerSequencePad) {
        EMIT(SequencePad, out(0), in(0), node.length(), node.value());
    } else if (node.op_type() == Node::kChainerSequenceRange) {
        EMIT(SequenceRange, out(0), in(0), oin(1), oin(2));
    } else if (node.op_type() == Node::kChainerGenericLen) {
        EMIT(GenericLen, out(0), in(0));
    } else if (node.op_type() == Node::kChainerGenericGetItem) {
        EMIT(GenericGetItem, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kChainerGenericGetSlice) {
        EMIT(GenericGetSlice, out(0), in(0), oin(1), oin(2), oin(3));
    } else if (node.op_type() == Node::kChainerGenericAdd) {
        EMIT(GenericAdd, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kChainerGenericAccumulateGrad) {
        EMIT(GenericAccumulateGrad, out(0), in(0), in(1));
    } else if (node.op_type() == Node::kChainerNullConstant) {
        EMIT(NullConstant, out(0));
    } else if (node.op_type() == Node::kWhere) {
        CHECK_EQ(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Where, out(0), in(0), in(1), in(2));
    } else if (node.op_type() == Node::kQuantizeLinear) {
        CHECK_LE(2UL, node.inputs().size());
        CHECK_GE(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(QuantizeLinear, out(0), in(0), in(1), oin(2));
    } else if (node.op_type() == Node::kDequantizeLinear) {
        CHECK_LE(2UL, node.inputs().size());
        CHECK_GE(3UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(DequantizeLinear, out(0), in(0), in(1), oin(2));
    } else if (node.op_type() == Node::kQLinearConv) {
        CHECK_LE(8UL, node.inputs().size());
        CHECK_GE(9UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        // TODO(ChainerX): Support dilation.
        for (int d : node.dilations()) CHECK_EQ(d, 1) << "Dilation is not supported yet";
        EMIT(QLinearConv,
             out(0),
             in(0),
             in(1),
             in(2),
             in(3),
             in(4),
             in(5),
             in(6),
             in(7),
             oin(8),
             strides(),
             pads(),
             node.group(),
             auto_pad());
    } else if (node.op_type() == Node::kMatMulInteger) {
        CHECK_LE(2UL, node.inputs().size());
        CHECK_GE(4UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(MatMulInteger, out(0), in(0), in(1), oin(2), oin(3));
    } else if (node.op_type() == Node::kConvInteger) {
        CHECK_LE(2UL, node.inputs().size());
        CHECK_GE(4UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(ConvInteger, out(0), in(0), in(1), oin(2), oin(3), strides(), pads(), node.group(), auto_pad());
    } else if (node.op_type() == Node::kBitShift) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(BitShift, out(0), in(0), in(1), node.direction());
    } else if (node.op_type() == Node::kNonZero) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(NonZero, out(0), in(0));
    } else if (node.op_type() == Node::kTopK) {
        CHECK_EQ(2UL, node.inputs().size());
        CHECK_EQ(2UL, node.outputs().size());
        EMIT(TopK, out(0), out(1), in(0), in(1), node.axis(), node.largest(), node.sorted());
    } else if (node.op_type() == Node::kNonMaxSuppression) {
        CHECK_LE(2UL, node.inputs().size());
        CHECK_GE(5UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(NonMaxSuppression, out(0), in(0), in(1), oin(2), oin(3), oin(4), node.center_point_box());
    } else if (node.op_type() == Node::kFlatten) {
        CHECK_EQ(1UL, node.inputs().size());
        CHECK_EQ(1UL, node.outputs().size());
        EMIT(Flatten, out(0), in(0), node.axis());
    } else if (node.op_type() == Node::kChainerBatchNormalizationExpandedStatsShape) {
        EMIT(BatchNormalizationExpandedStatsShape, out(0), in(0));
    } else if (node.op_type() == Node::kCumSum) {
        CHECK_EQ(0, node.exclusive());
        CHECK_EQ(0, node.reverse());
        EMIT(CumSum, out(0), in(0), oin(1));
    } else {
        CHECK(false) << "Unsupported op: " << node.op_type();
    }
}

}  // namespace chxvm
}  // namespace chainer_compiler
