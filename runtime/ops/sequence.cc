#include <xchainer/routines/creation.h>
#include <xchainer/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

void SequenceClearOp::RunImpl(XCVMState* st) {
    st->GetSequence(seq)->clear();
}

void SequenceAppendOp::RunImpl(XCVMState* st) {
    st->GetSequence(seq)->push_back(st->GetVar(value));
}

void SequenceLookupOp::RunImpl(XCVMState* st) {
    const std::vector<xchainer::Array>& v = *st->GetSequence(seq);
    CHECK_LT(index, v.size());
    st->SetVar(output, v[index]);
}

void SequenceStackOp::RunImpl(XCVMState* st) {
    const std::vector<xchainer::Array>& v = *st->GetSequence(seq);
    CHECK(!v.empty());
    std::vector<xchainer::Array> reshaped;
    for (const xchainer::Array& a : v) {
        xchainer::Shape shape{a.shape()};
        shape.insert(shape.begin(), 1);
        reshaped.push_back(xchainer::Reshape(a, shape));
    }

    const std::vector<xchainer::Array>& inputs = reshaped;
    const int axis = 0;
    // TODO(hamaji): Move this logic to xChainer.
    CHECK_LT(0, inputs.size());
    int64_t axis_dim = 0;
    for (const xchainer::Array& input : inputs) {
        CHECK_LT(axis, input.shape().size());
        CHECK_EQ(input.dtype(), inputs[0].dtype());
        CHECK_EQ(input.shape().size(), inputs[0].shape().size());
        for (int i = 0; i < input.shape().size(); ++i) {
            if (i != axis) CHECK_EQ(input.shape()[i], inputs[0].shape()[i]);
        }
        axis_dim += input.shape()[axis];
    }

    xchainer::Shape shape = inputs[0].shape();
    shape[axis] = axis_dim;
    // TODO(hamaji): Check why we cannot use `Empty` here.
    xchainer::Array result = xchainer::Zeros(shape, inputs[0].dtype(), inputs[0].device());
    std::vector<xchainer::ArrayIndex> indices(inputs[0].shape().size(), xchainer::Slice());
    axis_dim = 0;
    for (const xchainer::Array& input : inputs) {
        int64_t cur_dim = input.shape()[axis];
        indices[axis] = xchainer::Slice(axis_dim, axis_dim + cur_dim);
        result.At(indices) += input;
        axis_dim += cur_dim;
    }
    st->SetVar(output, result);
}

}  // namespace runtime
}  // namespace oniku
