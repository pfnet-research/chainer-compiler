#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xchainer.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

namespace {

class ConcatBackwardContext : public XCVMState::Auxiliary {
public:
    explicit ConcatBackwardContext(const std::vector<int64_t>& split) : split_(split) {
    }
    virtual ~ConcatBackwardContext() = default;

    const std::vector<int64_t>& split() {
        return split_;
    }

private:
    std::vector<int64_t> split_;
};

}  // namespace

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
    std::vector<nonstd::optional<chainerx::Array>>* v = st->GetSequence(seq);
    CHECK(!v->empty());
    st->SetVar(output, *v->back());
    v->pop_back();
    if (move_aux) {
        st->SetAux(output, st->PopAux(seq));
    }
}

void SequenceLookupOp::RunImpl(XCVMState* st) {
    const std::vector<nonstd::optional<chainerx::Array>>& v = *st->GetSequence(seq);
    int64_t i = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(index)));
    if (i < 0) i += v.size();
    CHECK_LT(i, v.size());
    st->SetVar(output, *v[i]);
}

void SequenceStackOp::RunImpl(XCVMState* st) {
    st->SetVar(output, Stack(NonOptional(*st->GetSequence(seq)), axis));
}

void SequenceConcatOp::RunImpl(XCVMState* st) {
    const std::vector<nonstd::optional<chainerx::Array>>& v = *st->GetSequence(seq);
    std::vector<int64_t> split;
    for (const nonstd::optional<chainerx::Array>& a : v) {
        split.push_back(a->shape()[axis]);
    }
    st->SetVar(output, Concat(NonOptional(v), axis));
    st->SetAux(output, std::shared_ptr<XCVMState::Auxiliary>(new ConcatBackwardContext(split)));
}

void SequenceConcatGradOp::RunImpl(XCVMState* st) {
    // TODO(hamaji): Probably better to get rid of auxiliary.
    auto ctx = dynamic_cast<ConcatBackwardContext*>(st->GetAux(y).get());
    CHECK(ctx);
    std::vector<nonstd::optional<chainerx::Array>>* seq = st->CreateSequence(gx);
    for (const chainerx::Array& a : Split(st->GetVar(gy), ctx->split(), axis)) {
        seq->push_back(a);
    }
}

void SequencePadOp::RunImpl(XCVMState* st) {
    const std::vector<nonstd::optional<chainerx::Array>>& v = *st->GetSequence(seq);
    CHECK(!v.empty());
    chainerx::Scalar p(value, v[0]->dtype());
    st->SetVar(output, PadSequence(NonOptional(v), length, p));
}

void SequenceRangeOp::RunImpl(XCVMState* st) {
    std::vector<nonstd::optional<chainerx::Array>>* seq = st->CreateSequence(output);
    int64_t start, stop, step = 1;
    if (arg1 >= 0) {
        start = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(arg0)));
        stop = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(arg1)));
        if (arg2 >= 0) {
            step = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(arg2)));
        }
    } else {
        start = 0;
        stop = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(arg0)));
    }
    CHECK_NE(step, 0);

    for (int64_t i = start; step > 0 ? (i < stop) : (i > stop); i += step) {
        seq->push_back(MakeArray(chainerx::Dtype::kInt64, {}, &i));
    }
}

namespace {

void SplitToSequence(chainerx::Array v, int axis, std::vector<nonstd::optional<chainerx::Array>>* seq) {
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
    std::vector<nonstd::optional<chainerx::Array>>* seq = st->CreateSequence(output);
    SplitToSequence(v, axis, seq);
}

void SequenceUnpadOp::RunImpl(XCVMState* st) {
    chainerx::Array v = st->GetVar(input);
    const std::vector<nonstd::optional<chainerx::Array>>& lens = *st->GetSequence(lengths);
    std::vector<nonstd::optional<chainerx::Array>>* seq = st->CreateSequence(output);
    SplitToSequence(v, 0, seq);
    for (size_t i = 0; i < seq->size(); ++i) {
        chainerx::ArrayIndex index = chainerx::Slice(0, int64_t(chainerx::AsScalar(*lens[i])));
        (*seq)[i] = (*seq)[i]->At({index});
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
    std::vector<nonstd::optional<chainerx::Array>>* lengths = st->CreateSequence(output);
    for (const nonstd::optional<chainerx::Array>& a : *st->GetSequence(seq)) {
        size_t len = a->shape()[0];
        lengths->push_back(MakeHostArray(chainerx::Dtype::kInt64, {}, &len));
    }
}

void SequenceCopyOp::RunImpl(XCVMState* st) {
    const std::vector<nonstd::optional<chainerx::Array>>& s = *st->GetSequence(seq);
    std::vector<nonstd::optional<chainerx::Array>>* d = st->CreateSequence(output);
    CHECK(d->empty());
    *d = s;
}

void SequenceMoveOp::RunImpl(XCVMState* st) {
    std::vector<nonstd::optional<chainerx::Array>>* s = st->GetSequence(seq);
    std::vector<nonstd::optional<chainerx::Array>>* d = st->CreateSequence(output);
    CHECK(d->empty());
    std::swap(*d, *s);
}

}  // namespace runtime
}  // namespace oniku
