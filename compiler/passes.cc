#include "passes.h"

#include <memory>

#include <compiler/flags.h>
#include <compiler/gradient.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/recompute.h>
#include <compiler/scheduler.h>
#include <compiler/simplifier.h>
#include <compiler/type_inference.h>

namespace oniku {

namespace {

void CollectGarbageNode(Graph* graph) {
    for (const auto& node : graph->nodes()) {
        if (node->onikux_order() <= 0) graph->DetachNode(node.get());
    }
}

template <class Fn>
void Recursively(Fn fn, Graph* graph) {
    fn(graph);
    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        if (node->body().get()) {
            fn(node->body().get());
        }
    }
}

}  //  namespace

void RunDefaultPasses(Model* model, bool gen_backprop) {
    Graph* graph = model->mutable_graph();
    InferAllDtypeAndShape(graph);

    Recursively(Simplify, graph);

    if (gen_backprop) AddGradientNodes(graph);
    if (g_recompute_relu) GetReluRecompute(graph, g_recompute_relu);

    Recursively([](Graph* g){ScheduleComputation(*g);}, graph);
    Recursively(CollectGarbageNode, graph);
}

}  // namespace oniku
