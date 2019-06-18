#include <set>

#include <compiler/fusion.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace chainer_compiler {

void FuseElementwiseOperations(Graph* graph) {
    // TODO(hamaji): Do not try fusing integer ops.
    const std::set<Node::OpType> fusable_ops = {
            Node::kIdentity,
            Node::kAdd,
            Node::kSub,
            Node::kMul,
            // Node::kDiv,
            Node::kTanh,
            Node::kSigmoid,
            Node::kExp,
    };

    auto is_fusable = [&fusable_ops](const Node& node) {
        if (node.op_type() == Node::kConstant) {
            Tensor* t = node.tensor_value().get();
            return t->dtype().IsFloat() && t->NumElements() == 1;
        }

        if (!fusable_ops.count(node.op_type())) return false;
        for (Value* value : node.inputs()) {
            Dtype dtype = value->type().dtype();
            // TODO(hamaji): Fix the dtype inference and do not fuse
            // unknown dtypes.
            if (!dtype.IsFloat() && dtype != Dtype::kUnknown) return false;
        }
        return true;
    };

    FuseAllConnectedNodes("nvrtc", graph, 2, false, is_fusable);
}

}  // namespace chainer_compiler
