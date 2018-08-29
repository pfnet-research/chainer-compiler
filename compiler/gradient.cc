#include "gradient.h"

#include <map>
#include <queue>
#include <set>

#include <onnx/onnx.pb.h>

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
        CHECK_EQ(1UL, graph->output_values().size());
    }

    void Run() {
        std::set<Value*> original_input_values;
        for (Value* input : graph_->GetNecessaryInputs()) {
            if (!input->initializer()) continue;
            CHECK(original_input_values.emplace(input).second);
        }

        for (Value* value : graph_->output_values()) {
            // TODO(hamaji): Refactor code to support non-float values.
            CHECK_EQ(Dtype::kFloat32, value->type().dtype());
            std::vector<float> data(1, 1.0);
            Value* one = graph_->AddConstValue("grad_in_one@" + value->name(), Type(value->type().dtype(), {}), data);
            Value* shape = graph_->AddValue("grad_in_shape@" + value->name());
            Value* grad = graph_->AddValue("grad_in@" + value->name());
            graph_->AddNode(Node::kShape, {value}, {shape});
            graph_->AddNode(Node::kExpand, {one, shape}, {grad});
            SetGrad(value, grad);
            op_queue_.push(value->producer());
        }

        while (!op_queue_.empty()) {
            const Node* node = op_queue_.front();
            CHECK(node);
            op_queue_.pop();
            if (!IsReady(node)) {
                op_queue_.push(node);
                continue;
            }
            if (!seen_nodes_.emplace(node).second) continue;

            AddGradientForNode(graph_, node);

            for (Value* input : node->inputs()) {
                if (input->producer()) op_queue_.push(input->producer());
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
            v->set_grad(nullptr);
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
            if (!value->grad()) return false;
        }
        return true;
    }

    void SetGrad(Value* y, Value* gy) {
        CHECK(y->grad() == nullptr);
        y->set_grad(gy);
    }

    Graph* graph_;
    std::queue<const Node*> op_queue_;
    std::set<const Node*> seen_nodes_;
};

}  // namespace

void AddGradientNodes(Graph* graph) {
    GradientGenerator gen(graph);
    gen.Run();
}

}  // namespace oniku
