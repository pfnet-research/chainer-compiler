#include "simplifier.h"

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {
namespace {

void RemoveSum(Graph* graph) {
    for (Node* node : graph->GetLiveNodes()) {
        if (node->op_type() != "Sum") continue;
        CHECK_EQ(1UL, node->outputs().size());
        Value* v = node->inputs()[0];
        for (size_t i = 1; i < node->inputs().size(); ++i) {
            Value* o = graph->AddValue(StrCat(node->name(), "_simplify_", i));
            graph->AddNode("Add", {v, node->inputs()[i]}, {o});
            v = o;
        }
        graph->AddNode("Ident", {v}, node->outputs());
        graph->DetachNode(node);
    }
}

void RemoveLess(Graph* graph) {
    for (Node* node : graph->GetLiveNodes()) {
        if (node->op_type() != "Less") continue;
        CHECK_EQ(2UL, node->inputs().size());
        CHECK_EQ(1UL, node->outputs().size());
        graph->AddNode("Greater",
                       {node->inputs()[1], node->inputs()[0]},
                       {node->outputs()[0]});
        graph->DetachNode(node);
    }
}

}  // namespace

void Simplify(Graph* graph) {
    RemoveSum(graph);
    RemoveLess(graph);
}

}  // namespace oniku
