#include <set>

#include <compiler/fusion.h>
#include <compiler/graph.h>

namespace chainer_compiler {

void FuseTensorRTOperations(Graph* graph) {
    // clang-format off
    static std::set<Node::OpType> fusable_ops = {
        Node::kAdd,
        Node::kAveragePool,
        Node::kBatchNormalization,
        Node::kConcat,
        Node::kConstant,
        Node::kConv,
        Node::kConvTranspose,
        Node::kIdentity,
        Node::kLeakyRelu,
        Node::kMaxPool,
        Node::kMul,
        Node::kNeg,
        Node::kReduceMean,
        Node::kReduceSum,
        Node::kRelu,
        Node::kSigmoid,
        Node::kSoftmax,
        Node::kSub,
        Node::kSum,
        Node::kTanh,
        Node::kTranspose,
        Node::kUnsqueeze
    };
    // clang-format on

    auto is_fusable = [](const Node& node) {
        if (!fusable_ops.count(node.op_type())) {
            return false;
        }
        for (Value* value : node.inputs()) {
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node.outputs()) {
            if (!value->type().HasKnownShape()) return false;
        }

        switch (node.op_type()) {
            case Node::kConvTranspose:
                // TODO(hamaji): Disabled, for now.
                return true;
            default:
                break;
        }

        return true;
    };

    FuseAllConnectedNodes("tensorrt", graph, 1, true, is_fusable);
}

}  // namespace chainer_compiler
