#include "gradient.h"

#include <queue>
#include <map>
#include <set>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/value.h>

namespace oniku {

namespace {

class GradientGenerator {
public:
    explicit GradientGenerator(Graph* graph)
        : graph_(graph) {
        CHECK_EQ(1UL, graph->output_values().size());
        for (Value* value : graph->output_values()) {
            // TODO(hamaji): Refactor code to support values other
            // than float32 loss.
#if 0
            CHECK(value->type().has_tensor_type()) << value->type().DebugString();
            CHECK(onnx::TensorProto::FLOAT == value->type().tensor_type().elem_type()) << value->type().DebugString();
            CHECK_EQ(1, value->type().tensor_type().shape().dim_size()) << value->type().DebugString();
#endif
            Value* grad = AddGradValue(value);
            onnx::TensorProto xtensor;
            // xtensor.set_data_type(value->type().tensor_type().elem_type());
            xtensor.set_data_type(onnx::TensorProto::FLOAT);
            xtensor.add_float_data(1.0);
            grad->ResetInitializer(std::make_unique<Tensor>(xtensor));
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
            if (!seen_nodes_.emplace(node).second)
                continue;

            AddGradient(node);

            for (Value* input : node->inputs()) {
                if (input->producer())
                    op_queue_.push(input->producer());
            }
        }

        for (Value* input : graph_->input_values()) {
            if (!input->initializer())
                continue;
            auto found = grad_values_.find(input);
            CHECK(found != grad_values_.end());
            Value* out_grad = graph_->AddValue("grad_out@" + input->name(), Value::Kind::kOutput);
            graph_->AddNode("Ident", {found->second}, {out_grad});
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
            if (!grad_values_.count(value))
                return false;
        }
        return true;
    }

    void AddGradient(const Node* node) {
        auto the_output = [&node]() {
            CHECK_EQ(1UL, node->outputs().size());
            return node->outputs()[0];
        };
        auto the_grad = [this, &node, &the_output]() {
            Value* value = the_output();
            auto found = grad_values_.find(value);
            CHECK(found != grad_values_.end());
            return found->second;
        };

        if (node->op_type() == "Add") {
            Value* gy = the_grad();
            for (Value* input : node->inputs())
                SetGrad(input, gy);
        } else if (node->op_type() == "Mul") {
            CHECK_EQ(2UL, node->inputs().size());
            Value* gy = the_grad();
            Value* x0 = node->inputs()[0];
            Value* x1 = node->inputs()[1];
            Value* gx0 = AddGradValue(x0);
            Value* gx1 = AddGradValue(x1);
            graph_->AddNode("Mul", {x1, gy}, {gx0});
            graph_->AddNode("Mul", {x0, gy}, {gx1});
        } else {
            CHECK(false) << "Gradient not supported: " << node->op_type();
        }
    }

    void SetGrad(Value* y, Value* gy) {
        CHECK(grad_values_.emplace(y, gy).second);
    }

    Graph* graph_;
    std::queue<const Node*> op_queue_;
    std::map<Value*, Value*> grad_values_;
    std::set<const Node*> seen_nodes_;
};

}  // namespace

void AddGradientNodes(Graph* graph) {
    GradientGenerator gen(graph);
    gen.Run();
}

}  // namespace oniku
