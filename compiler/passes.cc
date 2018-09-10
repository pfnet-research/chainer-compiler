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

void RunPassesInLoops(Graph* graph) {
    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        if (node->body().get()) {
            RunLoopBodyPasses(node->body().get());
        }
    }
}

}  //  namespace

void RunDefaultPasses(Model* model, bool gen_backprop) {
    Graph* graph = model->mutable_graph();
    InferAllDtypeAndShape(graph);
    Simplify(graph);
    if (gen_backprop) AddGradientNodes(graph);
    if (g_recompute_relu) GetReluRecompute(graph, g_recompute_relu);
    ScheduleComputation(*graph);
    RunPassesInLoops(graph);
    CollectGarbageNode(graph);
}

void RunLoopBodyPasses(Graph* graph) {
    // InferAllDtypeAndShape(graph);
    Simplify(graph);
    // if (gen_backprop) AddGradientNodes(graph);
    ScheduleComputation(*graph);
    RunPassesInLoops(graph);
    CollectGarbageNode(graph);
}

}  // namespace oniku
