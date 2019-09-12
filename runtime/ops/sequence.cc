#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/chxvm_state.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

int64_t GetOptionalInt(const absl::optional<StrictScalar>& array, int64_t default_value) {
    if (array.has_value()) {
        return static_cast<int64_t>(*array);
    } else {
        return default_value;
    }
}

}  // namespace

void SequenceClearOp::RunImpl(ChxVMState* st) {
    st->GetSequence(seq)->clear();
}

void SequenceAppendOp::RunImpl(ChxVMState* st) {
    st->GetSequence(seq)->emplace_back(*st->GetVar(value));
}

void SequenceInsertOp::RunImpl(
        ChxVMState* st, const ChxVMSequence& seq, const chainerx::Array& value, const StrictScalar& index, ChxVMSequence* output) {
    *output = seq;
    int64_t i = static_cast<int64_t>(index);
    if (i < 0) i += seq.size();
    CHECK_LT(i, seq.size());
    output->insert(output->begin() + i, ChxVMVar(value));
}

void SequenceExtendOp::RunImpl(ChxVMState* st, const ChxVMSequence& a, const ChxVMSequence& b, ChxVMSequence* output) {
    *output = a;
    for (const auto& a : b) output->push_back(a);
}

void SequencePopOp::RunImpl(ChxVMState* st) {
    // TODO(hamaji): Remove this code by removing null gradients.
    if (st->GetVar(seq)->IsNull()) {
        st->SetVar(output, ChxVMVar());
        return;
    }
    ChxVMSequence* v = st->GetSequence(seq);
    CHECK(!v->empty());
    st->SetVar(output, v->back());
    v->pop_back();
}

void SequenceEraseOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq, const StrictScalar& index, ChxVMSequence* output) {
    *output = seq;
    int64_t i = static_cast<int64_t>(index);
    if (i < 0) i += seq.size();
    CHECK_LT(i, seq.size());
    output->erase(output->begin() + i);
}

chainerx::Array SequenceLookupOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq, const StrictScalar& index) {
    int64_t i = static_cast<int64_t>(index);
    if (i < 0) i += seq.size();
    CHECK_LT(i, seq.size());
    return seq[i].GetArray();
}

void SequenceLookupGradOp::RunImpl(
        ChxVMState* st, const chainerx::Array& gy, const StrictScalar& size, const StrictScalar& index, ChxVMSequence* gx) {
    int64_t i = static_cast<int64_t>(index);
    int64_t sz = static_cast<int64_t>(size);
    if (i < 0) i += sz;
    CHECK_LT(i, sz);
    gx->resize(sz);
    (*gx)[i] = ChxVMVar(gy);
}

void SequenceUpdateOp::RunImpl(
        ChxVMState* st, const ChxVMSequence& seq, const StrictScalar& index, const chainerx::Array& value, ChxVMSequence* output) {
    *output = seq;
    int64_t i = static_cast<int64_t>(index);
    if (i < 0) i += seq.size();
    CHECK_LT(i, seq.size());
    (*output)[i] = ChxVMVar(value);
}

void SequenceGetSliceOp::RunImpl(
        ChxVMState* st,
        const ChxVMSequence& seq,
        const absl::optional<StrictScalar>& start_array,
        const absl::optional<StrictScalar>& end_array,
        const absl::optional<StrictScalar>& step_array,
        ChxVMSequence* output) {
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
        ChxVMState* st,
        const ChxVMSequence& gy,
        const chainerx::Array& size_array,
        const absl::optional<StrictScalar>& start_array,
        const absl::optional<StrictScalar>& end_array,
        const absl::optional<StrictScalar>& step_array,
        ChxVMSequence* gx) {
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

chainerx::Array SequenceStackOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq) {
    return chainerx::Stack(NonOptional(seq), axis);
}

std::tuple<chainerx::Array, chainerx::Array> SequenceConcatOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq) {
    const int ax = axis < 0 ? axis + seq.size() : axis;
    CHECK_GE(ax, 0);
    CHECK_LT(ax, seq.size());
    int64_t index = 0;
    std::vector<int64_t> indices;
    for (const ChxVMVar& v : seq) {
        indices.push_back(index += v.GetArray().shape()[ax]);
    }
    indices.pop_back();
    chainerx::Array out = chainerx::Concatenate(NonOptional(seq), ax);
    chainerx::Array ctx = MakeHostArray(chainerx::Dtype::kInt64, chainerx::Shape({static_cast<int64_t>(indices.size())}), &indices[0]);
    return std::tie(out, ctx);
}

void SequenceSplitAxisOp::RunImpl(
        ChxVMState* st, const chainerx::Array& seq, const absl::optional<chainerx::Array>& indices_or_sections, ChxVMSequence* output) {
    const int axis = ResolveAxis(seq, this->axis);
    if (!indices_or_sections.has_value() || indices_or_sections->ndim() == 0) {
        int64_t sections =
                indices_or_sections.has_value() ? static_cast<int64_t>(chainerx::AsScalar(*indices_or_sections)) : seq.shape()[axis];
        for (const chainerx::Array& a : chainerx::Split(seq, sections, axis)) {
            output->emplace_back(a);
        }
    } else {
        CHECK_EQ(1, indices_or_sections->ndim());
        std::vector<int64_t> indices;
        for (int i = 0; i < indices_or_sections->shape()[0]; ++i) {
            indices.push_back(static_cast<int64_t>(chainerx::AsScalar(indices_or_sections->At({i}))));
        }
        for (const chainerx::Array& a : chainerx::Split(seq, indices, axis)) {
            output->emplace_back(a);
        }
    }
}

chainerx::Array SequencePadOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq) {
    CHECK(!seq.empty());
    chainerx::Scalar p(value, chainerx::GetKind(seq[0].GetArray().dtype()));
    return PadSequence(NonOptional(seq), length, p);
}

void SequenceRangeOp::RunImpl(
        ChxVMState* st,
        const StrictScalar& arg0,
        const absl::optional<StrictScalar>& arg1,
        const absl::optional<StrictScalar>& arg2,
        ChxVMSequence* output) {
    int64_t start, stop, step = 1;
    if (arg1.has_value()) {
        start = static_cast<int64_t>(arg0);
        stop = static_cast<int64_t>(*arg1);
        if (arg2.has_value()) {
            step = static_cast<int64_t>(*arg2);
        }
    } else {
        start = 0;
        stop = static_cast<int64_t>(arg0);
    }
    CHECK_NE(step, 0);

    for (int64_t i = start; step > 0 ? (i < stop) : (i > stop); i += step) {
        output->emplace_back(MakeHostArray(chainerx::Dtype::kInt64, {}, &i));
    }
}

namespace {

void SplitToSequence(chainerx::Array v, int axis, ChxVMSequence* seq) {
    for (chainerx::Array a : chainerx::Split(v, v.shape()[axis], axis)) {
        chainerx::Shape shape{a.shape()};
        shape.erase(shape.begin() + axis);
        seq->emplace_back(chainerx::Reshape(a, shape));
    }
}

}  // namespace

void SequenceSeparateOp::RunImpl(ChxVMState* st, const chainerx::Array& input, ChxVMSequence* output) {
    const int axis = ResolveAxis(input, this->axis);
    SplitToSequence(input, axis, output);
}

void SequenceUnpadOp::RunImpl(ChxVMState* st, const chainerx::Array& input, const ChxVMSequence& lengths, ChxVMSequence* output) {
    SplitToSequence(input, 0, output);
    for (size_t i = 0; i < output->size(); ++i) {
        chainerx::ArrayIndex index = chainerx::Slice(0, int64_t(lengths[i].GetScalar()));
        (*output)[i] = ChxVMVar((*output)[i].GetArray().At({index}));
    }
}

void SequenceCreateOp::RunImpl(ChxVMState* st, const std::vector<chainerx::Array>& inputs, ChxVMSequence* output) {
    for (const chainerx::Array& a : inputs) {
        output->emplace_back(a);
    }
}

chainerx::Array SequenceSizeOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq) {
    int64_t size = seq.size();
    return MakeHostArray(chainerx::Dtype::kInt64, {}, &size);
}

void SequenceLengthsOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq, ChxVMSequence* output) {
    for (const ChxVMVar& v : seq) {
        size_t len = v.GetArray().shape()[0];
        output->emplace_back(MakeHostArray(chainerx::Dtype::kInt64, {}, &len));
    }
}

void SequenceCopyOp::RunImpl(ChxVMState* st, const ChxVMSequence& seq, ChxVMSequence* output) {
    *output = seq;
}

void SequenceMoveOp::RunImpl(ChxVMState* st) {
    // TODO(hamaji): Remove this code by removing null gradients.
    if (st->GetVar(seq)->IsNull()) {
        st->SetVar(output, ChxVMVar());
        return;
    }
    ChxVMSequence* s = st->GetSequence(seq);
    ChxVMSequence* d = st->CreateSequence(output);
    CHECK(d->empty());
    std::swap(*d, *s);
}

}  // namespace runtime
}  // namespace chainer_compiler
