#include "subgraph_canonicalizer.h"

#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/type.h>
#include <compiler/value.h>

namespace oniku {

namespace {

std::set<Value*> GetRequiredValues(Graph* graph) {
    std::set<Value*> required_values;
    for (Value* value : graph->output_values()) {
        required_values.insert(value);
    }
    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        for (Value* value : node->inputs()) {
            required_values.insert(value);
        }
    }

    for (Value* value : graph->input_values()) {
        required_values.erase(value);
    }
    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        for (Value* value : node->outputs()) {
            required_values.erase(value);
        }
    }
    return required_values;
}

void ResolveExternalDependencies(Graph* graph) {
    std::map<std::string, Value*> symtab;
    for (Value* value : graph->input_values()) {
        symtab[value->name()] = value;
    }
    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        for (Value* value : node->outputs()) {
            symtab[value->name()] = value;
        }
    }

    for (const std::unique_ptr<Node>& node : graph->nodes()) {
        if (node->op_type() == Node::kLoop) {
            Graph* body = node->body().get();
            CHECK(body);
            ResolveExternalDependencies(body);

            std::set<Value*> required_values = GetRequiredValues(node->body().get());
            for (Value* required : required_values) {
                Value* new_input = body->AddInputValue("CanonicalizeLoopBodyIn@" + required->name(), required->type());
                body->AddNode(Node::kIdentity, {new_input}, {required}, "CanonicalizeLoop");
                Value* new_output = body->AddOutputValue("CanonicalizeLoopBodyOut@" + required->name(), required->type());
                body->AddNode(Node::kIdentity, {new_input}, {new_output}, "CanonicalizeLoop");

                auto found = symtab.find(required->name());
                Value* external = nullptr;
                if (found == symtab.end()) {
                    external = graph->AddValue("CanonicalizeLoopIn@" + required->name());
                    external->set_type(new Type(required->type()));
                } else {
                    external = found->second;
                }
                node->AddInput(external);
                Value* dummy = graph->AddValue("CanonicalizeLoopUnusedOut@" + required->name());
                node->AddOutput(dummy);
            }
        } else if (node->op_type() == Node::kIf) {
            CHECK(node->then_branch().get());
            CHECK(node->else_branch().get());
            ResolveExternalDependencies(node->then_branch().get());
            ResolveExternalDependencies(node->else_branch().get());

            // TODO(hamaji): Implement.
        } else if (node->op_type() == Node::kOnikuxLoopRef) {
            // Do nothing. OnikuxLoopRef must not have external dependencies.
        } else {
            // Note `Scan` must be already removed.
            CHECK(!node->body().get());
            CHECK(!node->then_branch().get());
            CHECK(!node->else_branch().get());
        }
    }
}

}  // namespace

void CanonicalizeSubGraphs(Graph* graph) {
    ResolveExternalDependencies(graph);
}

}  // namespace oniku
