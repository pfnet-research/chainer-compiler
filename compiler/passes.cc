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

void ScheduleBackpropGraphs(Graph* graph) {
    struct SubGraph {
        Node* node;
        struct Ref {
            Node* node;
            const std::vector<std::string> input_value_names;
            const std::vector<std::string> output_value_names;
        };
        std::vector<Ref> refs;
    };

    std::map<Graph*, SubGraph> sub_graphs;
    for (Node* node : graph->nodes()) {
        for (Graph* sub_graph : node->GetSubGraphs()) {
            SubGraph sg;
            sg.node = node;
            CHECK(sub_graphs.emplace(sub_graph, sg).second);
        }
    }

    auto add_sub_graph_ref = [&sub_graphs, graph](Node* node, const std::string& graph_name, const std::vector<std::string>& input_value_names, const std::vector<std::string>& output_value_names) {
        if (graph_name.empty()) return;
        Graph* sub_graph = graph->GetSubGraph(graph_name);
        auto found = sub_graphs.find(sub_graph);
        CHECK(found != sub_graphs.end()) << graph_name;
        found->second.refs.emplace_back(SubGraph::Ref{node, input_value_names, output_value_names});
    };

    for (Node* node : graph->nodes()) {
        add_sub_graph_ref(node, node->body_ref(), node->input_value_names(), node->output_value_names());
        add_sub_graph_ref(node, node->then_branch_ref(), node->then_input_value_names(), node->then_output_value_names());
        add_sub_graph_ref(node, node->else_branch_ref(), node->else_input_value_names(), node->else_output_value_names());
    }

    for (const auto& p : sub_graphs) {
        Graph* graph = p.first;
        const SubGraph& sg = p.second;

        std::map<std::string, Value*> values;
        for (Value* v : graph->temp_values()) {
            CHECK(values.emplace(v->name(), v).second) << v->name();
        }

        for (const SubGraph::Ref& ref : sg.refs) {
            std::vector<Value*> input_values;
            for (const std::string& name : ref.input_value_names) {
                if (name.empty()) continue;
                auto found = values.find(name);
                CHECK(found != values.end()) << name;
                input_values.push_back(found->second);
            }
            std::vector<Value*> output_values;
            for (const std::string& name : ref.output_value_names) {
                if (name.empty()) continue;
                auto found = values.find(name);
                CHECK(found != values.end()) << name;
                output_values.push_back(found->second);
            }
            ScheduleComputation(*graph, input_values, output_values);
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

    Recursively([gen_backprop](Graph* g){ Simplify(g, gen_backprop); }, graph);

    CanonicalizeSubGraphs(graph);

    dump_onnx(g_dump_after_simplification, "after simplification");

    if (gen_backprop) AddGradientNodes(graph, g_always_retain_in_stack);

    Recursively([gen_backprop](Graph* g){ Simplify(g, gen_backprop); }, graph);

    dump_onnx(g_dump_after_gradient, "after gradient generation");

    if (g_dump_subgraphs) {
        graph->DumpSubGraphs();
    }

    Recursively(PropagateConstant, graph);

    if (g_recompute_relu) GetReluRecompute(graph, g_recompute_relu);

    if (g_fuse_operations) {
        Recursively(FuseOperations, graph);
        dump_onnx(g_dump_after_fusion, "after fusion");
    }

    Recursively([](Graph* g) { ScheduleComputation(*g); }, graph);
    if (gen_backprop) Recursively(ScheduleBackpropGraphs, graph);

    dump_onnx(g_dump_after_scheduling, "after scheduling");

    Recursively(CollectGarbageNode, graph);
}

}  // namespace oniku
