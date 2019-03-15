#include "compiler/shape_evaluator.h"

#include <string.h>

#include <vector>

#include <common/iterator.h>
#include <compiler/evaluator.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/value.h>

namespace chainer_compiler {

namespace {

bool HasKnownInputsAndUnknownOutputs(const Node& node) {
    bool has_unknown_outputs = false;
    for (Value* output : node.outputs()) {
        const Type& type = output->type();
        if (type.kind() == Type::Kind::kTensor && !type.HasKnownShape()) {
            has_unknown_outputs = true;
            break;
        }
    }
    if (!has_unknown_outputs) {
        return false;
    }

    if (node.inputs().empty()) {
        return false;
    }
    for (Value* input : node.inputs()) {
        if (!input->type().HasKnownShape()) {
            return false;
        }
    }
    return true;
}

bool MaybeEvaluateShape(Node* node) {
    switch (node->op_type()) {
        // TODO(hamaji): Handle more ops.
        case Node::kOneHot: {
            if (node->input(1)->producer()->op_type() == Node::kConstant) {
                DoEvaluateShape(node);
            }
            return true;
        }

        default:
            CLOG() << "Not propagate " << node->ToString() << std::endl;
    }
    return false;
}

}  // namespace

void DoEvaluateShape(Node* node) {
    CLOG() << "Evaluate shape of " << node->ToString() << std::endl;

    std::vector<std::unique_ptr<Tensor>> tensor_buf;
    std::vector<std::pair<Value*, Tensor*>> feeds;
    for (Value* input : node->inputs()) {
        Tensor* t;
        if (input->producer() && input->producer()->op_type() == Node::kConstant) {
            // To handle output shapes which depend on input values,
            // we use the constant value if `input` is a constant.
            // The caller of this function is responsible not to call
            // this function for this kind of ops.
            Node* const_node = input->producer();
            t = const_node->tensor_value().get();
            CHECK(t) << const_node->DebugString();
        } else {
            const Type& type = input->type();
            int64_t nbytes = type.GetNBytes();
            CHECK_LT(0, nbytes);
            Tensor::UniqueData data(std::malloc(nbytes), &std::free);
            memset(data.get(), 0, nbytes);
            CHECK_NE(Dtype::kUnknown, type.dtype()) << input->DebugString();
            t = new Tensor(input->name(), type.dtype(), type.dims(), data.release());
            tensor_buf.emplace_back(t);
        }
        feeds.emplace_back(input, t);
    }

    std::vector<std::unique_ptr<EvaluatedValue>> results;
    Eval({node}, feeds, node->outputs(), &results);

    CHECK_EQ(results.size(), node->outputs().size());
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        Value* value = node->output(i);
        if (value->type().HasKnownShape() || !result->is_tensor()) {
            continue;
        }
        std::unique_ptr<Tensor> r(result->ReleaseTensor());
        value->set_type(new Type(r->dtype(), r->dims()));
        CLOG() << " output #" << i << ": " << value->DebugString();
    }
}

void EvaluateShapes(Graph* graph) {
    bool replaced = true;
    while (replaced) {
        replaced = false;
        for (Node* node : graph->GetLiveNodes()) {
            if (!HasKnownInputsAndUnknownOutputs(*node)) {
                continue;
            }
            if (MaybeEvaluateShape(node)) {
                replaced = true;
            }
        }
    }
}

}  // namespace chainer_compiler
