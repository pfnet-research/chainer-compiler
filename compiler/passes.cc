#include "passes.h"

#include <map>
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

void ScheduleBackpropGraphs(Graph* graph) {
    struct SubGraph {
        Node* node;
        std::vector<Node*> refs;
    };

    std::map<Graph*, SubGraph> sub_graphs;
    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        if (node->body().get()) {
            SubGraph sg;
            sg.node = node.get();
            CHECK(sub_graphs.emplace(node->body().get(), sg).second);
        }
    }

    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        if (node->body_ref().empty()) continue;
        Graph* body = graph->GetSubGraph(node->body_ref());
        auto found = sub_graphs.find(body);
        CHECK(found != sub_graphs.end());
        found->second.refs.push_back(node.get());
    }

    for (const auto& p : sub_graphs) {
        Graph* graph = p.first;
        const SubGraph& sg = p.second;

        for (Node* ref : sg.refs) {
            std::map<std::string, Value*> values;
            for (Value* v : graph->temp_values()) {
                CHECK(values.emplace(v->name(), v).second);
            }
            std::vector<Value*> input_values;
            for (const std::string& name : ref->input_value_names()) {
                if (name.empty()) continue;
                auto found = values.find(name);
                CHECK(found != values.end()) << name;
                input_values.push_back(found->second);
            }
            std::vector<Value*> output_values;
            for (const std::string& name : ref->output_value_names()) {
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

    Recursively(Simplify, graph);

    if (gen_backprop) AddGradientNodes(graph);
    if (g_recompute_relu) GetReluRecompute(graph, g_recompute_relu);

    Recursively([](Graph* g){ScheduleComputation(*g);}, graph);
    if (gen_backprop) Recursively(ScheduleBackpropGraphs, graph);

    Recursively(CollectGarbageNode, graph);
}

}  // namespace oniku
