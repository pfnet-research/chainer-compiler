#include <set>

#include <compiler/fusion.h>
#include <compiler/graph.h>

namespace chainer_compiler {

void FuseSNPEOperations(Graph* graph) {
    static std::set<Node::OpType> fusable_ops = {
            // Node::kDropout,
            // Node::kAveragePool,
            Node::kMaxPool,
            Node::kBatchNormalization,
            Node::kConv,
            Node::kConvTranspose,
            Node::kGlobalAveragePool,
            Node::kGlobalMaxPool,
            Node::kMaxRoiPool,
            // TODO(take-cheeze): Fix typo in snpe-onnx-to-dlc
            // Node::kPRelu,
            Node::kLeakyRelu,
            Node::kLRN,
            Node::kAdd,
            Node::kArgMax,
            Node::kElu,
            Node::kGemm,
            Node::kMatMul,
            Node::kMax,
            Node::kMul,
            Node::kRelu,
            Node::kSigmoid,
            Node::kSoftmax,
            Node::kSum,
            // TODO(take-cheeze): Attribute bug in snpe-onnx-to-dlc
            // Node::kTanh,
            // TODO(take-cheeze): Removed in latest ONNX spec
            // Node::kScaledTanh,
            Node::kClip,
            Node::kConcat,
            Node::kConstant,
            Node::kFlatten,
            Node::kGather,
            Node::kPad,
            Node::kReshape,
            Node::kShape,
            Node::kSlice,
            Node::kSplit,
            Node::kSqueeze,
            Node::kTranspose,
            Node::kUnsqueeze,
            // Node::kUpsample,
            Node::kGRU,
            Node::kLSTM,
            Node::kRNN,
    };

    auto is_fusable = [](const Node& node) {
        if (!fusable_ops.count(node.op_type())) {
            return false;
        }

        switch (node.op_type()) {
            case Node::kMul:
            case Node::kAdd:
                if (!node.input(1)->initializer()) {
                    return false;
                }
                break;
            case Node::kMaxPool:
                if (node.auto_pad() != "NOTSET") {
                    return false;
                }
                break;
            case Node::kRNN:
                if (!node.input(1)->initializer()) {
                    return false;
                }
                break;
            case Node::kConv:
                if (!node.input(1)->initializer() || node.auto_pad() != "NOTSET") {
                    return false;
                }
                if (node.inputs().size() == 3 && !node.input(2)->initializer()) {
                    return false;
                }
                break;
            case Node::kMatMul:
                if (!node.input(1)->initializer()) {
                    return false;
                }
                break;
            case Node::kConvTranspose:
                if (!node.input(1)->initializer()) {
                    return false;
                }
                if (node.inputs().size() == 3 && !node.input(2)->initializer()) {
                    return false;
                }
                break;
            default:
                break;
        }

        for (Value* value : node.inputs()) {
            if (!value->type().HasKnownShape()) return false;
            if (value->name().find('@') != std::string::npos) return false;
        }
        for (Value* value : node.outputs()) {
            if (!value->type().HasKnownShape()) return false;
        }

        return true;
    };

    FuseAllConnectedNodes("snpe", graph, 1, true, is_fusable);
}

}  // namespace chainer_compiler
