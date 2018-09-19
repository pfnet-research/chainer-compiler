#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

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
    if (move_aux) {
        st->PushAux(seq, st->GetAux(value));
    }
}

void SequencePopOp::RunImpl(XCVMState* st) {
    std::vector<chainerx::Array>* v = st->GetSequence(seq);
    CHECK(!v->empty());
    st->SetVar(output, v->back());
    v->pop_back();
    if (move_aux) {
        st->SetAux(output, st->PopAux(seq));
    }
}

void SequenceLookupOp::RunImpl(XCVMState* st) {
    const std::vector<chainerx::Array>& v = *st->GetSequence(seq);
    int64_t i = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(index)));
    CHECK_LT(i, v.size());
    st->SetVar(output, v[i]);
}

void SequenceStackOp::RunImpl(XCVMState* st) {
    st->SetVar(output, Stack(*st->GetSequence(seq), axis));
}

void SequencePadOp::RunImpl(XCVMState* st) {
    const std::vector<chainerx::Array>& v = *st->GetSequence(seq);
    CHECK(!v.empty());
    chainerx::Scalar p(value, v[0].dtype());
    st->SetVar(output, PadSequence(v, length, p));
}

namespace {

void SplitToSequence(chainerx::Array v, int axis, std::vector<chainerx::Array>* seq) {
    std::vector<int64_t> lens(v.shape()[axis], 1);
    for (chainerx::Array a : Split(v, lens, axis)) {
        chainerx::Shape shape{a.shape()};
        shape.erase(shape.begin() + axis);
        seq->push_back(chainerx::Reshape(a, shape));
    }
}

}  // namespace

void SequenceSplitOp::RunImpl(XCVMState* st) {
    chainerx::Array v = st->GetVar(input);
    std::vector<chainerx::Array>* seq = st->CreateSequence(output);
    SplitToSequence(v, axis, seq);
}

void SequenceUnpadOp::RunImpl(XCVMState* st) {
    chainerx::Array v = st->GetVar(input);
    const std::vector<chainerx::Array>& lens = *st->GetSequence(lengths);
    std::vector<chainerx::Array>* seq = st->CreateSequence(output);
    SplitToSequence(v, 0, seq);
    for (size_t i = 0; i < seq->size(); ++i) {
        chainerx::ArrayIndex index = chainerx::Slice(0, int64_t(chainerx::AsScalar(lens[i])));
        (*seq)[i] = (*seq)[i].At({index});
    }
}

void SequenceCreateOp::RunImpl(XCVMState* st) {
    st->CreateSequence(output);
}

void SequenceSizeOp::RunImpl(XCVMState* st) {
    int64_t size = st->GetSequence(seq)->size();
    st->SetVar(output, MakeHostArray(chainerx::Dtype::kInt64, {}, &size));
}

void SequenceLengthsOp::RunImpl(XCVMState* st) {
    std::vector<chainerx::Array>* lengths = st->CreateSequence(output);
    for (chainerx::Array a : *st->GetSequence(seq)) {
        size_t len = a.shape()[0];
        lengths->push_back(MakeHostArray(chainerx::Dtype::kInt64, {}, &len));
    }
}

void SequenceCopyOp::RunImpl(XCVMState* st) {
    const std::vector<chainerx::Array>& s = *st->GetSequence(seq);
    std::vector<chainerx::Array>* d = st->CreateSequence(output);
    CHECK(d->empty());
    *d = s;
}

void SequenceMoveOp::RunImpl(XCVMState* st) {
    std::vector<chainerx::Array>* s = st->GetSequence(seq);
    std::vector<chainerx::Array>* d = st->CreateSequence(output);
    CHECK(d->empty());
    std::swap(*d, *s);
}

}  // namespace runtime
}  // namespace oniku
