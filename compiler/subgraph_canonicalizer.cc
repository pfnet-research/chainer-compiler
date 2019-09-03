#include "compiler/subgraph_canonicalizer.h"

#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/type.h>
#include <compiler/value.h>

namespace chainer_compiler {

namespace {

std::map<std::string, Value*> GetRequiredValues(Graph* graph) {
    std::map<std::string, Value*> required_values;
    for (Value* value : graph->output_values()) {
        if (value->IsNull()) continue;
        required_values[value->name()] = value;
    }
    for (const Node* node : graph->nodes()) {
        for (Value* value : node->inputs()) {
            if (value->IsNull()) continue;
            required_values[value->name()] = value;
        }
    }

    for (Value* value : graph->input_values()) {
        required_values.erase(value->name());
    }
    for (const Node* node : graph->nodes()) {
        for (Value* value : node->outputs()) {
            required_values.erase(value->name());
        }
    }
    return required_values;
}

void ResolveExternalDependencies(Graph* graph) {
    std::map<std::string, Value*> symtab;
    for (Value* value : graph->input_values()) {
        symtab[value->name()] = value;
    }
    for (const Node* node : graph->nodes()) {
        for (Value* value : node->outputs()) {
            symtab[value->name()] = value;
        }
    }

    for (Node* node : graph->nodes()) {
        if (node->op_type() == Node::kLoop) {
            Graph* body = node->body().get();
            CHECK(body);
            ResolveExternalDependencies(body);
            int index = body->input_values().size() - 2;

            for (auto& p : GetRequiredValues(node->body().get())) {
                Value* required = p.second;
                Value* new_input = body->AddInputValue("CanonicalizeLoopBodyIn@" + required->name(), required->type());
                body->AddNode(Node::kIdentity, {new_input}, {required}, "CanonicalizeLoop");
                Value* new_output = body->AddOutputValue("CanonicalizeLoopBodyOut@" + required->name(), required->type(), index + 1);
                body->AddNode(Node::kIdentity, {new_input}, {new_output}, "CanonicalizeLoop");

                auto found = symtab.find(required->name());
                Value* external = nullptr;
                if (found == symtab.end()) {
                    external = graph->AddValue(required->name());
                    external->set_type(new Type(required->type()));
                } else {
                    external = found->second;
                }
                node->AddInput(external);
                Value* dummy = graph->AddValue("CanonicalizeLoopUnusedOut@" + required->name());
                node->AddOutput(dummy, index);
                index++;
            }
        } else if (node->op_type() == Node::kIf) {
            Graph* then_branch = node->then_branch().get();
            Graph* else_branch = node->else_branch().get();
            CHECK(then_branch);
            CHECK(else_branch);
            ResolveExternalDependencies(then_branch);
            ResolveExternalDependencies(else_branch);

            std::map<std::string, Value*> required_values_then = GetRequiredValues(node->then_branch().get());
            std::map<std::string, Value*> required_values_else = GetRequiredValues(node->else_branch().get());
            std::map<std::string, Value*> required_values = required_values_then;
            required_values.insert(required_values_else.begin(), required_values_else.end());

            for (auto& p : required_values) {
                Value* required = p.second;
                {
                    Value* new_input = then_branch->AddInputValue("CanonicalizeIfThenIn@" + required->name(), required->type());
                    auto found = required_values_then.find(required->name());
                    if (found != required_values_then.end()) {
                        then_branch->AddNode(Node::kIdentity, {new_input}, {found->second}, "CanonicalizeIfThen");
                    }
                }
                {
                    Value* new_input = else_branch->AddInputValue("CanonicalizeIfElseIn@" + required->name(), required->type());
                    auto found = required_values_else.find(required->name());
                    if (found != required_values_else.end()) {
                        else_branch->AddNode(Node::kIdentity, {new_input}, {found->second}, "CanonicalizeIfElse");
                    }
                }

                auto found = symtab.find(required->name());
                Value* external = nullptr;
                if (found == symtab.end()) {
                    external = graph->AddValue(required->name());
                    external->set_type(new Type(required->type()));
                } else {
                    external = found->second;
                }
                node->AddInput(external);
            }
        } else if (node->op_type() == Node::kScan) {
            // Scan will be replaced by the simplifier.
        } else {
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

}  // namespace chainer_compiler
