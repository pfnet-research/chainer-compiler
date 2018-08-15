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
        for (Value* value : graph->output_values()) {
            // TODO(hamaji): Refactor code to support non-float values.
            CHECK_EQ(Dtype::kFloat32, value->type().dtype());
            Value* grad = graph_->AddInputValue("grad@" + value->name(), value->type());
            SetGrad(value, grad);
            init_grads_.emplace(grad);

            std::vector<float> data;
            for (int d : value->type().dims()) {
                CHECK_LT(0, d);
                for (int i = 0; i < d; ++i) data.push_back(1.0);
            }
            grad->ResetInitializer(std::make_unique<Tensor>(grad->name(), value->type().dtype(), value->type().dims(), data));
            op_queue_.push(value->producer());
        }
    }

    void Run() {
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
            if (!input->initializer()) continue;
            if (init_grads_.count(input)) continue;
            CHECK(input->grad());
            Value* out_grad = graph_->AddOutputValue("grad_out@" + input->name(), input->type());
            graph_->AddNode("Ident", {input->grad()}, {out_grad});
        }

        // Reset gradients.
        for (const auto& v : graph_->all_values()) {
            v->set_grad(nullptr);
        }
    }

private:
    Value* AddGradValue(Value* v) {
        Value* gv = graph_->AddValue("grad@" + v->name());
        SetGrad(v, gv);
        return gv;
    }

    bool IsReady(const Node* node) const {
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
    std::set<Value*> init_grads_;
};

}  // namespace

void AddGradientNodes(Graph* graph) {
    GradientGenerator gen(graph);
    gen.Run();
}

}  // namespace oniku
