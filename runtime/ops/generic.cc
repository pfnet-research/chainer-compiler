#include <numeric>

#include <chainerx/array.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/chxvm_state.h>
#include <runtime/chxvm_var.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

namespace {

int64_t GetSize(ChxVMVar* var) {
    switch (var->kind()) {
        case ChxVMVar::Kind::kArray:
            return var->GetArray().shape()[0];
        case ChxVMVar::Kind::kSequence:
            return var->GetSequence()->size();
        case ChxVMVar::Kind::kScalar:
        case ChxVMVar::Kind::kShape:
        case ChxVMVar::Kind::kOpaque:
        case ChxVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
    CHECK(false);
}

}  // namespace

void InOp::RunImpl(ChxVMState* st) {
    st->Input(name, v);
}

void OutOp::RunImpl(ChxVMState* st) {
    st->Output(name, v);
}

void IdentityOp::RunImpl(ChxVMState* st) {
    st->SetVar(y, *st->GetVar(x));
}

void FreeOp::RunImpl(ChxVMState* st) {
    st->FreeVar(v);
}

void PrintOp::RunImpl(ChxVMState* st) {
    for (int v : values) {
        std::cout << st->GetVar(v)->DebugString() << std::endl;
    }
}

void NullConstantOp::RunImpl(ChxVMState* st) {
    st->SetVar(output, ChxVMVar());
}

void GenericLenOp::RunImpl(ChxVMState* st) {
    ChxVMVar* var = st->GetVar(v);
    int64_t size = GetSize(var);
    st->SetArray(len, MakeHostArray(chainerx::Dtype::kInt64, {}, &size));
}

void GenericGetItemOp::RunImpl(ChxVMState* st) {
    ChxVMVar* var = st->GetVar(v);
    int64_t size = GetSize(var);
    int64_t i = static_cast<int64_t>(st->GetVar(index)->GetScalar());
    if (i < 0) i += size;
    switch (var->kind()) {
        case ChxVMVar::Kind::kShape: {
            CHECK_LT(i, var->GetShape().size());
            int64_t v = var->GetShape()[i];
            st->SetArray(output, MakeHostArray(chainerx::Dtype::kInt64, {}, &v));
            break;
        }

        case ChxVMVar::Kind::kArray:
            CHECK_LT(i, var->GetArray().shape()[0]);
            st->SetArray(output, var->GetArray().At({i}));
            break;

        case ChxVMVar::Kind::kSequence: {
            const ChxVMSequence& v = *var->GetSequence();
            CHECK_LT(i, v.size());
            st->SetArray(output, v[i].GetArray());
            break;
        }

        case ChxVMVar::Kind::kScalar:
        case ChxVMVar::Kind::kOpaque:
        case ChxVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
}

void GenericGetSliceOp::RunImpl(ChxVMState* st) {
    ChxVMVar* var = st->GetVar(v);
    int64_t size = GetSize(var);
    int64_t start = st->GetOptionalInt(this->start, 0);
    if (start < 0) start += size;
    start = std::max<int64_t>(0, start);
    start = std::min<int64_t>(size, start);
    int64_t end = st->GetOptionalInt(this->end, size);
    if (end < 0) end += size;
    end = std::max<int64_t>(0, end);
    end = std::min<int64_t>(size, end);
    int64_t step = st->GetOptionalInt(this->step, 1);
    CHECK_NE(0, step) << "Slice step cannot be zero";

    switch (var->kind()) {
        case ChxVMVar::Kind::kArray: {
            st->SetArray(output, var->GetArray().At({chainerx::Slice(start, end, step)}));
            break;
        }

        case ChxVMVar::Kind::kSequence: {
            const ChxVMSequence& v = *var->GetSequence();
            ChxVMSequence* seq = st->CreateSequence(output);
            if (step > 0) {
                for (int64_t i = start; i < end; i += step) {
                    CHECK_LE(0, i);
                    CHECK_LT(i, v.size());
                    seq->push_back(v[i]);
                }
            } else {
                for (int64_t i = start; i > end; i += step) {
                    CHECK_LE(0, i);
                    CHECK_LT(i, v.size());
                    seq->push_back(v[i]);
                }
            }
            break;
        }

        case ChxVMVar::Kind::kScalar:
        case ChxVMVar::Kind::kShape:
        case ChxVMVar::Kind::kOpaque:
        case ChxVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
}

void GenericAddOp::RunImpl(ChxVMState* st) {
    ChxVMVar* var0 = st->GetVar(a);
    ChxVMVar* var1 = st->GetVar(b);

    if (var0->IsArray() || var1->IsArray()) {
        auto to_a = [](ChxVMVar* v) {
            if (v->IsArray()) {
                return v->GetArray();
            }
            return Stack(NonOptional(*v->GetSequence()), 0);
        };
        st->SetArray(output, to_a(var0) + to_a(var1));
    } else {
        ChxVMSequence* seq = st->CreateSequence(output);
        *seq = *var0->GetSequence();
        for (const auto& a : *var1->GetSequence()) seq->push_back(a);
    }
}

void GenericIsOp::RunImpl(ChxVMState* st) {
    ChxVMVar* var0 = st->GetVar(a);
    ChxVMVar* var1 = st->GetVar(b);

    bool result = false;
    if (var0->kind() != var1->kind()) {
        // We are sure the return value is false.
    } else if (var0->IsArray()) {
        chainerx::Array a = var0->GetArray();
        chainerx::Array b = var1->GetArray();
        if (a.ndim() == 0 && b.ndim() == 0 && a.dtype() == chainerx::Dtype::kBool && b.dtype() == chainerx::Dtype::kBool) {
            result = (static_cast<bool>(chainerx::AsScalar(a)) == static_cast<bool>(chainerx::AsScalar(b)));
        } else if (a.dtype() != b.dtype() || a.shape() != b.shape() || a.raw_data() != b.raw_data()) {
            // We are sure the return value is false.
        } else {
            WARN_ONCE("`is` keyword for non boolean scalars might be wrong");
        }
    } else {
        WARN_ONCE("`is` keyword for sequences might be wrong");
    }
    st->SetArray(output, MakeHostArray(chainerx::Dtype::kBool, {}, &result));
}

void GenericAccumulateGradOp::RunImpl(ChxVMState* st) {
    ChxVMVar* var0 = st->GetVar(a);
    ChxVMVar* var1 = st->GetVar(b);

    // TODO(hamaji): Add testcases which require these checks.
    if (var0->kind() == ChxVMVar::Kind::kNull) {
        st->SetVar(output, *var1);
        return;
    }
    if (var1->kind() == ChxVMVar::Kind::kNull) {
        st->SetVar(output, *var0);
        return;
    }
    CHECK_EQ(var0->kind(), var1->kind()) << var0->DebugString() << " vs " << var1->DebugString();

    switch (var0->kind()) {
        case ChxVMVar::Kind::kArray: {
            st->SetArray(output, var0->GetArray() + var1->GetArray());
            break;
        }
        case ChxVMVar::Kind::kSequence: {
            const ChxVMSequence& seq0 = *var0->GetSequence();
            const ChxVMSequence& seq1 = *var1->GetSequence();
            ChxVMSequence* out = st->CreateSequence(output);
            CHECK_EQ(seq0.size(), seq1.size()) << var0->DebugString() << " vs " << var1->DebugString();
            out->resize(seq0.size());
            for (size_t i = 0; i < seq0.size(); ++i) {
                if (!seq0[i].IsNull() && !seq1[i].IsNull()) {
                    (*out)[i] = ChxVMVar(seq0[i].GetArray() + seq1[i].GetArray());
                } else if (!seq0[i].IsNull()) {
                    (*out)[i] = seq0[i];
                } else if (!seq1[i].IsNull()) {
                    (*out)[i] = seq1[i];
                }
            }
            break;
        }

        case ChxVMVar::Kind::kScalar:
        case ChxVMVar::Kind::kShape:
        case ChxVMVar::Kind::kOpaque:
        case ChxVMVar::Kind::kNull:
            CHECK(false) << var0->DebugString();
    }
}

}  // namespace runtime
}  // namespace chainer_compiler
