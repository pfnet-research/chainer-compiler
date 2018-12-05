#include "passes.h"

#include <map>
#include <memory>

#include <compiler/constant_propagation.h>
#include <compiler/context.h>
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

void CheckAllOpsSupported(const CompilerContext& cctx, Graph* graph) {
    for (Node* node : graph->nodes()) {
        CHECK(cctx.HasOp(node->op_type())) << "Op not supported by backend (" << cctx.name() << ")\n" << node->DebugString();
    }
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
    RunDefaultPasses(model->mutable_graph(), gen_backprop);
}

void RunDefaultPasses(Graph* graph, bool gen_backprop) {
    std::unique_ptr<CompilerContext> cctx{GetCompilerContext(g_backend_name)};

    InferAllDtypeAndShape(graph);

    auto dump_onnx = [&graph](bool cond, const char* msg) {
        if (cond) {
            std::cerr << "=== vvv " << msg << " vvv ===\n";
            std::cerr << graph->DebugString();
            std::cerr << "=== ^^^ " << msg << " ^^^ ===\n";
        }
        Recursively([msg](Graph* g) { g->CheckSanity(msg); }, graph);
    };

    dump_onnx(g_dump_after_inference, "after inference");

    CanonicalizeSubGraphs(graph);

    Recursively([&cctx, gen_backprop](Graph* g) { Simplify(*cctx, g, gen_backprop); }, graph);

    Recursively(PropagateConstants, graph);

    Recursively([](Graph* g) { g->DeleteDetached(); }, graph);

    dump_onnx(g_dump_after_simplification, "after simplification");

    if (gen_backprop) AddGradientNodesForTraining(graph);

    // TODO(hamaji): Make it possible to infer shapes here.
    // if (!g_skip_inference) graph->InferShapes();

    Recursively([&cctx, gen_backprop](Graph* g) { Simplify(*cctx, g, gen_backprop); }, graph);

    Recursively(PropagateConstants, graph);

    Recursively([](Graph* g) { g->DeleteDetached(); }, graph);

    dump_onnx(g_dump_after_gradient, "after gradient generation");

    if (g_dump_subgraphs) {
        graph->DumpSubGraphs();
    }

    if (g_recompute_relu) GetReluRecompute(graph, g_recompute_relu);

    if (g_fuse_operations) {
        Recursively(FuseOperations, graph);
        dump_onnx(g_dump_after_fusion, "after fusion");
    }

    int64_t order = 0;
    Recursively([&order](Graph* g) { order = ScheduleComputation(*g, order); }, graph);

    dump_onnx(g_dump_after_scheduling, "after scheduling");

    Recursively(CollectGarbageNode, graph);

    Recursively([&cctx](Graph* g) { CheckAllOpsSupported(*cctx, g); }, graph);
}

void RunDefaultPassesBeforeGradient(Graph* graph) {
    std::unique_ptr<CompilerContext> cctx{GetCompilerContext(g_backend_name)};
    graph->InferShapes();
    CanonicalizeSubGraphs(graph);
    Recursively([&cctx](Graph* g) { Simplify(*cctx, g, true); }, graph);
    Recursively(PropagateConstants, graph);
    Recursively([](Graph* g) { g->DeleteDetached(); }, graph);
    Recursively([&cctx](Graph* g) { CheckAllOpsSupported(*cctx, g); }, graph);
}

}  // namespace oniku
