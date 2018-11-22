#include "passes.h"

#include <map>
#include <memory>

#include <compiler/constant_propagation.h>
#include <compiler/flags.h>
#include <compiler/fusion.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/recompute.h>
#include <compiler/scheduler.h>
#include <compiler/simplifier.h>
#include <compiler/subgraph_canonicalizer.h>
#include <compiler/type_inference.h>

namespace oniku {

namespace {

void CollectGarbageNode(Graph* graph) {
    for (Node* node : graph->nodes()) {
        if (node->onikux_order() <= 0) graph->DetachNode(node);
    }
    graph->DeleteDetached();
}

template <class Fn>
void Recursively(Fn fn, Graph* graph) {
    fn(graph);
    for (const Node* node : graph->nodes()) {
        for (Graph* subgraph : node->GetSubGraphs()) {
            Recursively(fn, subgraph);
        }
    }
}

}  //  namespace

void RunDefaultPasses(Model* model, bool gen_backprop) {
    Graph* graph = model->mutable_graph();
    InferAllDtypeAndShape(graph);

    auto dump_onnx = [&graph](bool cond, const char* msg) {
        if (cond) {
            std::cerr << "=== vvv " << msg << " vvv ===\n";
            std::cerr << graph->DebugString();
            std::cerr << "=== ^^^ " << msg << " ^^^ ===\n";
        }
    };

    dump_onnx(g_dump_after_inference, "after inference");

    Recursively([gen_backprop](Graph* g) { Simplify(g, gen_backprop); }, graph);

    CanonicalizeSubGraphs(graph);

    Recursively(PropagateConstants, graph);

    dump_onnx(g_dump_after_simplification, "after simplification");

    if (gen_backprop) AddGradientNodesForTraining(graph);

    Recursively([gen_backprop](Graph* g) { Simplify(g, gen_backprop); }, graph);

    dump_onnx(g_dump_after_gradient, "after gradient generation");

    if (g_dump_subgraphs) {
        graph->DumpSubGraphs();
    }

    Recursively(PropagateConstants, graph);

    if (g_recompute_relu) GetReluRecompute(graph, g_recompute_relu);

    if (g_fuse_operations) {
        Recursively(FuseOperations, graph);
        dump_onnx(g_dump_after_fusion, "after fusion");
    }

    int64_t order = 0;
    Recursively([&order](Graph* g) { order = ScheduleComputation(*g, order); }, graph);

    dump_onnx(g_dump_after_scheduling, "after scheduling");

    Recursively(CollectGarbageNode, graph);
}

}  // namespace oniku
