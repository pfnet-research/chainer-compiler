#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xcvm_state.h>

namespace oniku {
namespace runtime {

namespace {

int64_t GetOptionalInt(const nonstd::optional<chainerx::Array>& array, int64_t default_value) {
    if (array.has_value()) {
        return static_cast<int64_t>(chainerx::AsScalar(*array));
    } else {
        return default_value;
    }
}

}  // namespace

void SequenceClearOp::RunImpl(XCVMState* st) {
    st->GetSequence(seq)->clear();
}

void SequenceAppendOp::RunImpl(XCVMState* st) {
    st->GetSequence(seq)->emplace_back(*st->GetVar(value));
}

void SequencePopOp::RunImpl(XCVMState* st) {
    // TODO(hamaji): Remove this code by removing null gradients.
    if (st->GetVar(seq)->IsNull()) {
        st->SetVar(output, XCVMVar());
        return;
    }
    XCVMSequence* v = st->GetSequence(seq);
    CHECK(!v->empty());
    st->SetVar(output, v->back());
    v->pop_back();
}

chainerx::Array SequenceLookupOp::RunImpl(XCVMState* st, const XCVMSequence& seq, const chainerx::Array& index) {
    int64_t i = static_cast<int64_t>(chainerx::AsScalar(index));
    if (i < 0) i += seq.size();
    CHECK_LT(i, seq.size());
    return seq[i].GetArray();
}

void SequenceLookupGradOp::RunImpl(
        XCVMState* st, const chainerx::Array& gy, const chainerx::Array& size, const chainerx::Array& index, XCVMSequence* gx) {
    int64_t i = static_cast<int64_t>(chainerx::AsScalar(index));
    int64_t sz = static_cast<int64_t>(chainerx::AsScalar(size));
    if (i < 0) i += sz;
    CHECK_LT(i, sz);
    gx->resize(sz);
    (*gx)[i] = XCVMVar(gy);
}

void SequenceGetSliceOp::RunImpl(
        XCVMState* st,
        const XCVMSequence& seq,
        const nonstd::optional<chainerx::Array>& start_array,
        const nonstd::optional<chainerx::Array>& end_array,
        const nonstd::optional<chainerx::Array>& step_array,
        XCVMSequence* output) {
    int64_t size = seq.size();
    int64_t start = GetOptionalInt(start_array, 0);
    if (start < 0) start += size;
    start = std::max<int64_t>(0, start);
    start = std::min<int64_t>(size, start);
    int64_t end = GetOptionalInt(end_array, size);
    if (end < 0) end += size;
    end = std::max<int64_t>(0, end);
    end = std::min<int64_t>(size, end);
    int64_t step = GetOptionalInt(step_array, 1);
    CHECK_NE(0, step) << "Slice step cannot be zero";

    for (int64_t i = start; step > 0 ? (i < end) : (i > end); i += step) {
        CHECK_LE(0, i);
        CHECK_LT(i, seq.size());
        output->push_back(seq[i]);
    }
}

void SequenceGetSliceGradOp::RunImpl(
        XCVMState* st,
        const XCVMSequence& gy,
        const chainerx::Array& size_array,
        const nonstd::optional<chainerx::Array>& start_array,
        const nonstd::optional<chainerx::Array>& end_array,
        const nonstd::optional<chainerx::Array>& step_array,
        XCVMSequence* gx) {
    int64_t size = static_cast<int64_t>(chainerx::AsScalar(st->GetArray(this->size)));
    int64_t start = GetOptionalInt(start_array, 0);
    if (start < 0) start += size;
    start = std::max<int64_t>(0, start);
    start = std::min<int64_t>(size, start);
    int64_t end = GetOptionalInt(end_array, size);
    if (end < 0) end += size;
    end = std::max<int64_t>(0, end);
    end = std::min<int64_t>(size, end);
    int64_t step = GetOptionalInt(step_array, 1);
    CHECK_NE(0, step) << "Slice step cannot be zero";

    gx->resize(size);
    int64_t j = 0;
    for (int64_t i = start; step > 0 ? (i < end) : (i > end); i += step) {
        (*gx)[i] = gy[j++];
    }
}

chainerx::Array SequenceStackOp::RunImpl(XCVMState* st, const XCVMSequence& seq) {
    return chainerx::Stack(NonOptional(seq), axis);
}

std::tuple<chainerx::Array, chainerx::Array> SequenceConcatOp::RunImpl(XCVMState* st, const XCVMSequence& seq) {
    int64_t index = 0;
    std::vector<int64_t> indices;
    for (const XCVMVar& v : seq) {
        indices.push_back(index += v.GetArray().shape()[axis]);
    }
    indices.pop_back();
    chainerx::Array out = chainerx::Concatenate(NonOptional(seq), axis);
    chainerx::Array ctx = MakeHostArray(chainerx::Dtype::kInt64, chainerx::Shape({static_cast<int64_t>(indices.size())}), &indices[0]);
    return std::tie(out, ctx);
}

void SequenceSplitAxisOp::RunImpl(
        XCVMState* st, const chainerx::Array& seq, const chainerx::Array& indices_or_sections, XCVMSequence* output) {
    if (indices_or_sections.ndim() == 0) {
        int64_t sections = static_cast<int64_t>(chainerx::AsScalar(indices_or_sections));
        for (const chainerx::Array& a : chainerx::Split(seq, sections, axis)) {
            output->emplace_back(a);
        }
    } else {
        CHECK_EQ(1, indices_or_sections.ndim());
        std::vector<int64_t> indices;
        for (int i = 0; i < indices_or_sections.shape()[0]; ++i) {
            indices.push_back(static_cast<int64_t>(chainerx::AsScalar(indices_or_sections.At({i}))));
        }
        for (const chainerx::Array& a : chainerx::Split(seq, indices, axis)) {
            output->emplace_back(a);
        }
    }
}

chainerx::Array SequencePadOp::RunImpl(XCVMState* st, const XCVMSequence& seq) {
    CHECK(!seq.empty());
    chainerx::Scalar p(value, seq[0].GetArray().dtype());
    return PadSequence(NonOptional(seq), length, p);
}

void SequenceRangeOp::RunImpl(
        XCVMState* st,
        const chainerx::Array& arg0,
        const nonstd::optional<chainerx::Array>& arg1,
        const nonstd::optional<chainerx::Array>& arg2,
        XCVMSequence* output) {
    int64_t start, stop, step = 1;
    if (arg1.has_value()) {
        start = static_cast<int64_t>(chainerx::AsScalar(arg0));
        stop = static_cast<int64_t>(chainerx::AsScalar(*arg1));
        if (arg2.has_value()) {
            step = static_cast<int64_t>(chainerx::AsScalar(*arg2));
        }
    } else {
        start = 0;
        stop = static_cast<int64_t>(chainerx::AsScalar(arg0));
    }
    CHECK_NE(step, 0);

    for (int64_t i = start; step > 0 ? (i < stop) : (i > stop); i += step) {
        output->emplace_back(MakeHostArray(chainerx::Dtype::kInt64, {}, &i));
    }
}

namespace {

void SplitToSequence(chainerx::Array v, int axis, XCVMSequence* seq) {
    for (chainerx::Array a : chainerx::Split(v, v.shape()[axis], axis)) {
        chainerx::Shape shape{a.shape()};
        shape.erase(shape.begin() + axis);
        seq->emplace_back(chainerx::Reshape(a, shape));
    }
}

}  // namespace

void SequenceSeparateOp::RunImpl(XCVMState* st, const chainerx::Array& input, XCVMSequence* output) {
    SplitToSequence(input, axis, output);
}

void SequenceUnpadOp::RunImpl(XCVMState* st, const chainerx::Array& input, const XCVMSequence& lengths, XCVMSequence* output) {
    SplitToSequence(input, 0, output);
    for (size_t i = 0; i < output->size(); ++i) {
        chainerx::ArrayIndex index = chainerx::Slice(0, int64_t(chainerx::AsScalar(lengths[i].GetArray())));
        (*output)[i] = XCVMVar((*output)[i].GetArray().At({index}));
    }
}

void SequenceCreateOp::RunImpl(XCVMState* st, XCVMSequence* output) {
}

chainerx::Array SequenceSizeOp::RunImpl(XCVMState* st, const XCVMSequence& seq) {
    int64_t size = seq.size();
    return MakeHostArray(chainerx::Dtype::kInt64, {}, &size);
}

void SequenceLengthsOp::RunImpl(XCVMState* st, const XCVMSequence& seq, XCVMSequence* output) {
    for (const XCVMVar& v : seq) {
        size_t len = v.GetArray().shape()[0];
        output->emplace_back(MakeHostArray(chainerx::Dtype::kInt64, {}, &len));
    }
}

void SequenceCopyOp::RunImpl(XCVMState* st, const XCVMSequence& seq, XCVMSequence* output) {
    *output = seq;
}

void SequenceMoveOp::RunImpl(XCVMState* st) {
    // TODO(hamaji): Remove this code by removing null gradients.
    if (st->GetVar(seq)->IsNull()) {
        st->SetVar(output, XCVMVar());
        return;
    }
    XCVMSequence* s = st->GetSequence(seq);
    XCVMSequence* d = st->CreateSequence(output);
    CHECK(d->empty());
    std::swap(*d, *s);
}

}  // namespace runtime
}  // namespace oniku
