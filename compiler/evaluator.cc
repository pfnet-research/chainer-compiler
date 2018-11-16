#include "evaluator.h"

#include <stdlib.h>
#include <string.h>

#include <set>

#include <chainerx/array.h>

#include <compiler/log.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/xcvm_emitter.h>
#include <compiler/xcvm_emitter.h>
#include <compiler/value.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_state.h>

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

void Eval(const std::vector<Node*>& nodes, const std::vector<Value*>& fetches, std::vector<std::unique_ptr<Tensor>>* outputs) {
    runtime::XCProgramProto program;
    std::vector<int> output_ids;
    xcvm::Emit(nodes, fetches, &program, &output_ids);
    // LOG() << "Evaluate " << program.DebugString();

    runtime::XCVM xcvm(program);
    runtime::XCVMState state(runtime::XCVMOptions{}, xcvm.num_variables(), {});
    xcvm.Run(&state);

    for (size_t i = 0; i < fetches.size(); ++i) {
        int output_id = output_ids[i];
        // TODO(hamaji): Support other types.
        outputs->emplace_back(ArrayToTensor(fetches[i]->name(), state.GetArray(output_id)));
    }
}

}  // namespace oniku
