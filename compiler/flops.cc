#include "compiler/flops.h"

#include <iostream>

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
    int64_t const out_size = node.output(0)->type().NumElements();
    int64_t flops = out_size;
    flops *= node.input(0)->type().dims()[node.trans_a() ? 0 : 1];
    if (node.alpha() != 1.0) {
        flops += out_size;
    }
    if (node.beta() != 0.0) {
        flops += out_size;
    }
    return flops;
}

int64_t LowerDimNumElements(const std::vector<int64_t>& dims, size_t start_index) {
    const size_t end_index = dims.size() - 1;
    int64_t ret = 1;
    for (size_t i = start_index; i <= end_index; ++i) {
        ret *= dims[i];
    }
    return ret;
}

int64_t CalculateFlopsOfConv(const Node& node) {
    Type const& x = node.input(0)->type();
    Type const& w = node.input(1)->type();
    Type const& y = node.output(0)->type();
    int64_t bsize = x.dims()[0];
    int64_t ichan = x.dims()[1];
    int64_t ochan = y.dims()[1];
    int64_t kernel_nums = LowerDimNumElements(w.dims(), 2);
    int64_t output_nums = LowerDimNumElements(y.dims(), 2);
    return (bsize * ichan * ochan * output_nums * kernel_nums) / node.group();
}

int64_t CalculateFlopsOfConvTranspose(Node const& node) {
    Type const& x = node.input(0)->type();
    Type const& w = node.input(1)->type();
    Type const& y = node.output(0)->type();
    int64_t const bsize = x.dims()[0];
    int64_t const ichan = x.dims()[1];
    int64_t const ochan = y.dims()[1];
    int64_t const kernel_nums = LowerDimNumElements(w.dims(), 2);
    int64_t const input_nums = LowerDimNumElements(x.dims(), 2);
    return (bsize * ichan * ochan * kernel_nums * input_nums) / node.group();
}

int64_t CalculateFlopsOfConvGradWeight(Node const& node) {
    Type const& w = node.input(0)->type();
    Type const& x = node.input(1)->type();
    Type const& gy = node.input(2)->type();

    CHECK_EQ(gy.dims()[0], x.dims()[0]);

    int64_t const bsize = x.dims()[0];
    int64_t const ic = x.dims()[1];
    int64_t const oc = gy.dims()[1];
    int64_t const input_nums = LowerDimNumElements(x.dims(), 2);
    int64_t const kernel_nums = LowerDimNumElements(w.dims(), 2);
    return (bsize * input_nums * oc * ic * kernel_nums) / node.group();
}

int64_t CalculateFlopsOfSoftmax(Node const& node) {
    int axis = node.axis();
    if (node.axis() < 0) {
        axis = node.input(0)->type().ndim() + axis;
    }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, node.input(0)->type().ndim());

    int64_t const c = node.input(0)->type().dims()[axis];
    int64_t const s = node.input(0)->type().NumElements() / c;
    return 2 * OutputSize(node) + s * (c - 1);
}

int64_t CalculateFlopsOfAveragePool(Node const& node) {
    int64_t const kw = node.kernel_shape()[0];
    int64_t const kh = node.kernel_shape()[1];
    return OutputSize(node) * kw * kh;
}

int64_t CalculateFlopsOfMaxPool(Node const& node) {
    int64_t const kw = node.kernel_shape()[0];
    int64_t const kh = node.kernel_shape()[1];
    return OutputSize(node) * (kw * kh - 1);
}

int64_t CalculateFlopsOfMax(Node const& node) {
    return (node.inputs().size() - 1) * OutputSize(node);
}

int64_t CalculateFlopsImpl(const Node& node) {
    if (node.IsZeroCost()) {
        return 0;
    }

    if (!HasKnownInOuts(node)) {
        return -1;
    }

    switch (node.op_type()) {
        case Node::kGemm:
            return CalculateFlopsOfGemm(node);

        case Node::kChainerFusionGroup:
            CHECK(false);

        case Node::kSum:
        case Node::kMin:
        case Node::kMax:
            return CalculateFlopsOfMax(node);

        case Node::kClip:
            return 2 * OutputSize(node);

        // Convolution nodes:
        case Node::kConv:
            return CalculateFlopsOfConv(node);

        case Node::kConvTranspose:
            return CalculateFlopsOfConvTranspose(node);

        case Node::kChainerConvGradWeight:
            return CalculateFlopsOfConvGradWeight(node);

        // Activation nodes:
        case Node::kSigmoid:
            return 4 * OutputSize(node);

        case Node::kLeakyRelu:
            return 2 * OutputSize(node);

        case Node::kSoftmax:
            return CalculateFlopsOfSoftmax(node);

        // Pooling nodes:
        case Node::kAveragePool:
            return CalculateFlopsOfAveragePool(node);

        case Node::kMaxPool:
            return CalculateFlopsOfMaxPool(node);

        default:
            return OutputSize(node);
    }
}

}  // namespace

int64_t CalculateFlops(const Node& node, int* num_unknown_flops) {
    std::vector<Graph*> subgraphs = node.GetSubGraphs();
    if (subgraphs.empty()) {
        int64_t flops = CalculateFlopsImpl(node);
        if (flops < 0 && num_unknown_flops) {
            // std::cerr << "Unknown flops: " << node.ToString() << std::endl;
            ++*num_unknown_flops;
        }
        return flops;
    }

    if (node.op_type() == Node::kChainerFusionGroup) {
        CHECK_EQ(1, subgraphs.size());
        return CalculateTotalFlops(*subgraphs[0], num_unknown_flops);
    } else {
        if (num_unknown_flops) {
            ++*num_unknown_flops;
        }
        return -1;
    }
}

int64_t CalculateTotalFlops(const Graph& graph, int* num_unknown_flops) {
    int64_t total_flops = 0;
    for (const Node* node : graph.GetComputationSequence()) {
        int64_t flops = CalculateFlops(*node, num_unknown_flops);
        // std::cerr << node->ToString() << " " << flops << std::endl;
        if (flops >= 0) {
            total_flops += flops;
        }
    }
    return total_flops;
}

void ShowFlops(const Graph& graph) {
    int num_unknown_flops = 0;
    int64_t total_flops = CalculateTotalFlops(graph, &num_unknown_flops);
    if (num_unknown_flops) {
        std::cerr << "Incomplete flops clalculation" << std::endl;
    }
    std::cerr << "Total flops: " << total_flops << std::endl;
}

}  // namespace chainer_compiler
