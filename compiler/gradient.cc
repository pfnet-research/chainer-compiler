#include "gradient.h"

#include <map>
#include <queue>
#include <set>

#include <onnx/onnx-ml.pb.h>

#include <common/log.h>
#include <compiler/gradient_ops.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>
#include <compiler/value.h>

namespace oniku {

namespace {

class GradientGenerator {
public:
    explicit GradientGenerator(Graph* graph) : graph_(graph) {
    }

    void Run(const std::vector<Value*>& ys) {
        necessary_values_ = graph_->GetNecessaryValues();
        std::set<Value*> original_input_values;
        for (Value* value : necessary_values_) {
            if (value->kind() != Value::Kind::kInput || !value->initializer()) continue;
            CHECK(original_input_values.emplace(value).second);
        }

        for (Value* y : ys) {
            CHECK(y->grad());
            op_queue_.push(y->producer());
        }

        int not_ready_count = 0;
        while (!op_queue_.empty()) {
            Node* node = op_queue_.front();
            CHECK(node);
            op_queue_.pop();
            if (!IsReady(node)) {
                op_queue_.push(node);
                if (++not_ready_count > op_queue_.size() * 2) {
                    std::cerr << "Infinite loop during backprop!" << std::endl;
                    while (!op_queue_.empty()) {
                        Node* node = op_queue_.front();
                        op_queue_.pop();
                        std::cerr << node->DebugString() << std::endl;
                    }
                    CHECK(false);
                }
                continue;
            }
            not_ready_count = 0;
            if (!seen_nodes_.emplace(node).second) continue;

            AddGradientForNode(graph_, node);

            for (Value* input : node->inputs()) {
                if (input->grad() && input->producer())
                    op_queue_.push(input->producer());
            }
        }

        for (Value* input : graph_->input_values()) {
            if (!original_input_values.count(input)) continue;
            if (!input->type().dtype().IsFloat()) continue;
            CHECK(input->grad()) << input->name();
            Value* out_grad = graph_->AddOutputValue("grad_out@" + input->name(), input->type());
            graph_->AddNode(Node::kIdentity, {input->grad()}, {out_grad});
        }

        // Reset gradients.
        for (const auto& v : graph_->all_values()) {
            if (Value* gv = v->grad()) {
                gv->set_type(new Type(v->type()));
                v->set_grad(nullptr);
            }
        }
    }

private:
    bool IsReady(const Node* node) const {
        // TODO(hamaji): Figure out a better way to select outputs
        // required to compute gradients.
        if (node->op_type() == Node::kBatchNormalization) {
            return node->outputs()[0]->grad();
        }
        for (Value* value : node->outputs()) {
            if (necessary_values_.count(value) && !value->grad()) return false;
        }
        return true;
    }

    Graph* graph_;
    std::queue<Node*> op_queue_;
    std::set<const Node*> seen_nodes_;
    std::set<Value*> necessary_values_;
};

}  // namespace

void AddGradientNodes(Graph* graph, const std::vector<Value*>& ys) {
    GradientGenerator gen(graph);
    gen.Run(ys);
}

void AddGradientNodes(Graph* graph) {
    CHECK_EQ(1UL, graph->output_values().size());
    std::vector<Value*> ys;
    for (Value* value : graph->output_values()) {
        // TODO(hamaji): Refactor code to support non-float values.
        CHECK_EQ(Dtype::kFloat32, value->type().dtype());
        std::vector<float> data(1, 1.0);
        Value* one = graph->AddConstValue("grad_in_one@" + value->name(), Type(value->type().dtype(), {}), data);
        Value* shape = graph->AddValue("grad_in_shape@" + value->name());
        Value* grad = graph->AddValue("grad_in@" + value->name());
        graph->AddNode(Node::kShape, {value}, {shape});
        graph->AddNode(Node::kExpand, {one, shape}, {grad});
        CHECK(value->grad() == nullptr);
        value->set_grad(grad);
        ys.push_back(value);
    }
    AddGradientNodes(graph, ys);
}

}  // namespace oniku
