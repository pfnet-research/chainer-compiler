#include "compiler/flops.h"

#include <compiler/graph.h>
#include <compiler/node.h>

namespace chainer_compiler {

namespace {

bool HasKnownInOuts(const Node& node) {
    for (Value* value : node.inputs()) {
        if (!value->type().HasKnownShape()) {
            return false;
        }
    }
    for (Value* value : node.outputs()) {
        if (!value->type().HasKnownShape()) {
            return false;
        }
    }
    return true;
}

int64_t OutputSize(const Node& node) {
    if (node.outputs().size() != 1) {
        return -1;
    }
    CHECK_EQ(1, node.outputs().size());
    return node.output(0)->type().NumElements();
}

int64_t CalculateFlopsOfGemm(const Node& node) {
    int64_t flops = node.output(0)->type().NumElements();
    flops *= node.input(0)->type().dims()[node.trans_a() ? 0 : 1];
    return flops;
}

int64_t CalculateFlopsOfConv(const Node& node) {
    int64_t bsize = node.input(0)->type().dims()[0];
    int64_t ichan = node.input(0)->type().dims()[1];
    int64_t kw = node.input(1)->type().dims()[2];
    int64_t kh = node.input(1)->type().dims()[3];
    int64_t ochan = node.output(0)->type().dims()[1];
    int64_t ow = node.output(0)->type().dims()[2];
    int64_t oh = node.output(0)->type().dims()[3];
    return bsize * ichan * ochan * ow * oh * kw * kh;
}

}

int64_t CalculateFlops(const Node& node) {
    if (!HasKnownInOuts(node)) {
        return -1;
    }
    switch (node.op_type()) {
        case Node::kGemm:
            return CalculateFlopsOfGemm(node);

        case Node::kConv:
            return CalculateFlopsOfConv(node);

        default:
            return OutputSize(node);
    }
}

void ShowFlops(const Graph& graph) {
    int64_t total_flops = 0;
    int num_unknown_flops = 0;
    for (const Node* node : graph.GetComputationSequence()) {
        int64_t flops = CalculateFlops(*node);
        if (flops < 0) {
            ++num_unknown_flops;
        } else {
            total_flops += flops;
        }
    }
    if (num_unknown_flops) {
        std::cerr << "Incomplete flops clalculation" << std::endl;
    }
    std::cerr << "Total flops: " << total_flops << std::endl;
}

}  // namespace chainer_compiler
