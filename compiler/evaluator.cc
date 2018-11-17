#include "evaluator.h"

#include <stdlib.h>
#include <string.h>

#include <set>

#include <chainerx/array.h>

#include <common/strutil.h>
#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/xcvm_emitter.h>
#include <compiler/xcvm_emitter.h>
#include <compiler/value.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_state.h>
#include <runtime/xcvm_var.h>

namespace oniku {

namespace {

Dtype GetDtype(const chainerx::Array& a) {
    switch (a.dtype()) {
    case chainerx::Dtype::kBool: return Dtype::kBool;
    case chainerx::Dtype::kInt8: return Dtype::kInt8;
    case chainerx::Dtype::kInt16: return Dtype::kInt16;
    case chainerx::Dtype::kInt32: return Dtype::kInt32;
    case chainerx::Dtype::kInt64: return Dtype::kInt64;
    case chainerx::Dtype::kUInt8: return Dtype::kUInt8;
    case chainerx::Dtype::kFloat32: return Dtype::kFloat32;
    case chainerx::Dtype::kFloat64: return Dtype::kFloat64;
    }
    CHECK(false) << a;
}

Tensor* ArrayToTensor(const std::string& name, const chainerx::Array& a) {
    Tensor::UniqueData data(std::malloc(a.GetNBytes()), &std::free);
    memcpy(data.get(), a.ToNative().raw_data(), a.GetNBytes());
    std::vector<int64_t> dims{a.shape().begin(), a.shape().end()};
    return new Tensor(name, GetDtype(a), dims, std::move(data));
}

}  // namespace

EvaluatedValue::EvaluatedValue(Tensor* tensor)
    : tensor_(tensor) {}

EvaluatedValue::EvaluatedValue(std::vector<std::unique_ptr<Tensor>>&& sequence)
    : sequence_(std::move(sequence)) {}

void Eval(const std::vector<Node*>& nodes, const std::vector<Value*>& fetches, std::vector<std::unique_ptr<EvaluatedValue>>* outputs) {
    runtime::XCProgramProto program;
    std::vector<int> output_ids;
    xcvm::Emit(nodes, fetches, &program, &output_ids);
    // LOG() << "Evaluate " << program.DebugString();

    runtime::XCVM xcvm(program);
    runtime::XCVMState state(runtime::XCVMOptions{}, xcvm.num_variables(), {});
    xcvm.Run(&state);

    for (size_t i = 0; i < fetches.size(); ++i) {
        const std::string& name = fetches[i]->name();
        int output_id = output_ids[i];
        runtime::XCVMVar* var = state.GetVar(output_id);

        switch (var->kind()) {
        case runtime::XCVMVar::Kind::kArray: {
            outputs->emplace_back(new EvaluatedValue(ArrayToTensor(name, state.GetArray(output_id))));
            break;
        }

        case runtime::XCVMVar::Kind::kSequence: {
            const runtime::XCVMSequence& seq = *var->GetSequence();
            std::vector<std::unique_ptr<Tensor>> tensors;
            for (size_t j = 0; j < seq.size(); ++j) {
                // TODO(hamaji): Support nested sequences.
                tensors.emplace_back(ArrayToTensor(StrCat(name, '_', j), seq[j].GetArray()));
            }
            outputs->emplace_back(new EvaluatedValue(std::move(tensors)));
            break;
        }

        default:
            // TODO(hamaji): Support other types.
            CHECK(false) << "Not supported yet: " << var->DebugString();
        }
    }
}

}  // namespace oniku
