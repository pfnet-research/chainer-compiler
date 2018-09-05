#include <xchainer/routines/creation.h>
#include <xchainer/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xchainer.h>
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
    int64_t i = static_cast<int64_t>(xchainer::AsScalar(st->GetVar(index)));
    CHECK_LT(i, v.size());
    st->SetVar(output, v[i]);
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
    st->SetVar(output, Concat(reshaped, 0));
}

void SequenceCreateOp::RunImpl(XCVMState* st) {
    st->CreateSequence(output);
}

void SequenceCopyOp::RunImpl(XCVMState* st) {
    const std::vector<xchainer::Array>& s = *st->GetSequence(seq);
    std::vector<xchainer::Array>* d = st->CreateSequence(output);
    CHECK(d->empty());
    *d = s;
}

void SequenceMoveOp::RunImpl(XCVMState* st) {
    std::vector<xchainer::Array>* s = st->GetSequence(seq);
    std::vector<xchainer::Array>* d = st->CreateSequence(output);
    CHECK(d->empty());
    std::swap(*d, *s);
}

}  // namespace runtime
}  // namespace oniku
