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
    return bsize * ichan * ochan * ow * oh * kw * kh / node.group();
}

int64_t CalculateFlopsImpl(const Node& node) {
    if (!HasKnownInOuts(node)) {
        return -1;
    }
    switch (node.op_type()) {
        case Node::kGemm:
            return CalculateFlopsOfGemm(node);

        case Node::kConv:
            return CalculateFlopsOfConv(node);

        case Node::kChainerFusionGroup:
            CHECK(false);

        default:
            return OutputSize(node);
    }
}

int64_t CalculateFlopsOfGraph(const Graph& graph, int* num_unknown_flops) {
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

}  // namespace

int64_t CalculateFlops(const Node& node, int* num_unknown_flops) {
    std::vector<Graph*> subgraphs = node.GetSubGraphs();
    if (subgraphs.empty()) {
        int64_t flops = CalculateFlopsImpl(node);
        if (flops < 0 && num_unknown_flops) {
            ++*num_unknown_flops;
        }
        return flops;
    }

    if (node.op_type() == Node::kChainerFusionGroup) {
        CHECK_EQ(1, subgraphs.size());
        return CalculateFlopsOfGraph(*subgraphs[0], num_unknown_flops);
    } else {
        ++*num_unknown_flops;
        return -1;
    }
}

void ShowFlops(const Graph& graph) {
    int num_unknown_flops = 0;
    int64_t total_flops = CalculateFlopsOfGraph(graph, &num_unknown_flops);
    if (num_unknown_flops) {
        std::cerr << "Incomplete flops clalculation" << std::endl;
    }
    std::cerr << "Total flops: " << total_flops << std::endl;
}

}  // namespace chainer_compiler
