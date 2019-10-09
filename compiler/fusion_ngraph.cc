#include <set>

#include <compiler/fusion.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

void FuseNGraphOperations(Graph* graph) {
    // TODO(hamaji): Enable all ops.
    const std::set<Node::OpType> fusable_ops = {
            Node::kAbs,
            Node::kAcos,
            Node::kAcosh,
            Node::kAdd,
            Node::kAnd,
            Node::kArgMax,
            Node::kArgMin,
            Node::kAsin,
            Node::kAsinh,
            Node::kAtan,
            Node::kAtanh,
            Node::kAveragePool,
            Node::kBatchNormalization,
            Node::kCeil,
            Node::kClip,
            Node::kConcat,
            Node::kConstant,
            Node::kConv,
            Node::kConvTranspose,
            Node::kCos,
            Node::kCosh,
            Node::kDiv,
            Node::kDropout,
            Node::kElu,
            Node::kEqual,
            Node::kExp,
            Node::kFlatten,
            Node::kFloor,
            Node::kGemm,
            Node::kGlobalAveragePool,
            // Not supported yet.
            // Node::kGlobalLpPool,
            Node::kGlobalMaxPool,
            Node::kGreater,
            // Not supported yet.
            // Node::kHardSigmoid,
            Node::kIdentity,
            // There seem to be some restrictions:
            // terminate called after throwing an instance of 'ngraph::NodeValidationFailure'
            // what():  Check '(input_shape.rank().is_dynamic() || static_cast<size_t>(input_shape.rank()) >= 3)'
            // Node::kLRN,
            Node::kLeakyRelu,
            Node::kLess,
            Node::kLog,
            Node::kLogSoftmax,
            Node::kMatMul,
            Node::kMax,
            Node::kMaxPool,
            Node::kMean,
            Node::kMin,
            Node::kMul,
            Node::kNeg,
            Node::kNot,
            // Constant input only.
            // Node::kOneHot,
            Node::kOr,
            // Not supported yet.
            // Node::kPRelu,
            Node::kPow,
            Node::kReciprocal,
            Node::kReduceL1,
            Node::kReduceL2,
            Node::kReduceLogSum,
            Node::kReduceLogSumExp,
            Node::kReduceMax,
            Node::kReduceMean,
            Node::kReduceMin,
            // Not supported yet.
            // Node::kReduceProd,
            Node::kReduceSum,
            Node::kReduceSumSquare,
            Node::kRelu,
            Node::kSelu,
            Node::kShape,
            Node::kSigmoid,
            Node::kSign,
            Node::kSin,
            Node::kSinh,
            Node::kSize,
            Node::kSlice,
            Node::kSoftmax,
            Node::kSoftplus,
            Node::kSoftsign,
            Node::kSplit,
            Node::kSqrt,
            Node::kSqueeze,
            Node::kSub,
            Node::kSum,
            Node::kTan,
            Node::kTanh,
            // Not supported yet.
            // Node::kTopK,
            Node::kTranspose,
            Node::kUnsqueeze,
            Node::kXor,
            Node::kWhere,
            Node::kPad,
            Node::kReshape,
    };

    const std::set<Node::OpType> negative_axes_ops = {
            Node::kArgMax,
            Node::kArgMin,
            Node::kFlatten,
            Node::kReduceL1,
            Node::kReduceL2,
            Node::kReduceLogSum,
            Node::kReduceLogSumExp,
            Node::kReduceMax,
            Node::kReduceMean,
            Node::kReduceMin,
            // Not supported yet.
            // Node::kReduceProd,
            Node::kReduceSum,
            Node::kReduceSumSquare,
            Node::kSqueeze,
            Node::kUnsqueeze,
    };

    auto is_fusable = [&fusable_ops, &negative_axes_ops](const Node& node) {
        if (!fusable_ops.count(node.op_type())) {
            return false;
        }
        for (Value* value : node.inputs()) {
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node.outputs()) {
            if (!value->type().HasKnownShape()) return false;
        }

        if (node.op_type() == Node::kReshape) {
            CHECK_EQ(2, node.inputs().size());
            if (!node.input(1)->producer() || node.input(1)->producer()->op_type() != Node::kConstant) {
                return false;
            }
        } else if (node.op_type() == Node::kMaxPool) {
            if (node.ceil_mode()) {
                return false;
            }
        } else if (
                node.op_type() == Node::kAdd || node.op_type() == Node::kSub || node.op_type() == Node::kMul ||
                node.op_type() == Node::kDiv || node.op_type() == Node::kPow) {
            // No type coercion in nGraph.
            if (node.input(0)->type().dtype() != node.input(1)->type().dtype()) {
                return false;
            }
        } else if (node.op_type() == Node::kPad) {
            // Apparently, nGraph does not support negative pads.
            for (int p : node.pads()) {
                if (p < 0) {
                    return false;
                }
            }
        } else if (node.op_type() == Node::kTranspose) {
            // Incomplete transpose is our own extension to ONNX.
            if (!node.perm().empty() && node.input(0)->type().ndim() != node.perm().size()) {
                return false;
            }
        } else if (node.op_type() == Node::kBatchNormalization) {
            // nGraph does not support BatchNorm in training mode.
            if (node.outputs().size() != 1) {
                return false;
            }
        } else if (node.op_type() == Node::kSlice) {
            // nGraph does not support new slice.
            if (node.inputs().size() > 1) {
                return false;
            }
        } else if (node.op_type() == Node::kSoftmax || node.op_type() == Node::kLogSoftmax) {
            // nGraph does not know Chainer's Softmax.
            if (!node.chainer_is_onnx_semantics()) {
                return false;
            }
        } else if (node.op_type() == Node::kClip || node.op_type() == Node::kPad) {
            // nGraph does not support Clip-11 nor Pad-11.
            if (node.inputs().size() > 1) {
                return false;
            }
        } else if (negative_axes_ops.count(node.op_type())) {
            // nGraph does not support negative axes in opset11.
            if (node.axis() < 0) {
                return false;
            }
            for (int axis : node.axes()) {
                if (axis < 0) {
                    return false;
                }
            }
        }

        return true;
    };

    FuseAllConnectedNodes("ngraph", graph, 1, true, is_fusable);
}

}  // namespace chainer_compiler
