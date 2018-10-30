#include <numeric>

#include <chainerx/array.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/gen_xcvm_ops.h>
#include <runtime/xchainer.h>
#include <runtime/xcvm_state.h>
#include <runtime/xcvm_var.h>

namespace oniku {
namespace runtime {

namespace {

int64_t GetSize(XCVMVar* var) {
    switch (var->kind()) {
        case XCVMVar::Kind::kArray:
            return var->GetArray().shape()[0];
        case XCVMVar::Kind::kSequence:
            return var->GetSequence()->size();
        case XCVMVar::Kind::kOpaque:
        case XCVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
    CHECK(false);
}

int64_t GetOptionalInt(XCVMState* st, int index, int64_t default_value) {
    nonstd::optional<chainerx::Array> var = st->GetOptionalArray(index);
    if (var.has_value()) {
        return static_cast<int64_t>(chainerx::AsScalar(*var));
    } else {
        return default_value;
    }
}

}  // namespace

void InOp::RunImpl(XCVMState* st) {
    st->Input(name, v);
}

void OutOp::RunImpl(XCVMState* st) {
    st->Output(name, v);
}

void IdentityOp::RunImpl(XCVMState* st) {
    XCVMVar* var = st->GetXCVMVar(x);
    switch (var->kind()) {
        case XCVMVar::Kind::kArray:
            st->SetArray(y, var->GetArray());
            break;

        case XCVMVar::Kind::kSequence: {
            const XCVMSequence& s = *var->GetSequence();
            XCVMSequence* d = st->CreateSequence(y);
            CHECK(d->empty());
            *d = s;
            break;
        }

        case XCVMVar::Kind::kOpaque:
        case XCVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
}

void FreeOp::RunImpl(XCVMState* st) {
    st->FreeVar(v);
}

void PrintOp::RunImpl(XCVMState* st) {
    for (int v : values) {
        std::cout << st->GetXCVMVar(v)->DebugString() << std::endl;
    }
}

void GenericLenOp::RunImpl(XCVMState* st) {
    XCVMVar* var = st->GetXCVMVar(v);
    int64_t size = GetSize(var);
    st->SetArray(len, MakeHostArray(chainerx::Dtype::kInt64, {}, &size));
}

void GenericGetItemOp::RunImpl(XCVMState* st) {
    XCVMVar* var = st->GetXCVMVar(v);
    int64_t size = GetSize(var);
    int64_t i = static_cast<int64_t>(chainerx::AsScalar(st->GetArray(index)));
    if (i < 0) i += size;
    switch (var->kind()) {
        case XCVMVar::Kind::kArray:
            CHECK_LT(i, var->GetArray().shape()[0]);
            st->SetArray(output, var->GetArray().At({i}));
            break;

        case XCVMVar::Kind::kSequence: {
            const XCVMSequence& v = *var->GetSequence();
            CHECK_LT(i, v.size());
            st->SetArray(output, v[i].GetArray());
            break;
        }

        case XCVMVar::Kind::kOpaque:
        case XCVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
}

void GenericGetSliceOp::RunImpl(XCVMState* st) {
    XCVMVar* var = st->GetXCVMVar(v);
    int64_t size = GetSize(var);
    int64_t start = GetOptionalInt(st, this->start, 0);
    if (start < 0) start += size;
    start = std::max<int64_t>(0, start);
    start = std::min<int64_t>(size, start);
    int64_t end = GetOptionalInt(st, this->end, size);
    if (end < 0) end += size;
    end = std::max<int64_t>(0, end);
    end = std::min<int64_t>(size, end);
    int64_t step = GetOptionalInt(st, this->step, 1);
    CHECK_NE(0, step) << "Slice step cannot be zero";

    switch (var->kind()) {
        case XCVMVar::Kind::kArray: {
            st->SetArray(output, var->GetArray().At({chainerx::Slice(start, end, step)}));
            break;
        }

        case XCVMVar::Kind::kSequence: {
            const XCVMSequence& v = *var->GetSequence();
            XCVMSequence* seq = st->CreateSequence(output);
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

        case XCVMVar::Kind::kOpaque:
        case XCVMVar::Kind::kNull:
            CHECK(false) << var->DebugString();
    }
}

void GenericAddOp::RunImpl(XCVMState* st) {
    XCVMVar* var0 = st->GetXCVMVar(a);
    XCVMVar* var1 = st->GetXCVMVar(b);

    if (var0->kind() == XCVMVar::Kind::kArray || var1->kind() == XCVMVar::Kind::kArray) {
        auto to_a = [](XCVMVar* v) {
            if (v->kind() == XCVMVar::Kind::kArray) {
                return v->GetArray();
            }
            return Stack(NonOptional(*v->GetSequence()), 0);
        };
        st->SetArray(output, to_a(var0) + to_a(var1));
    } else {
        XCVMSequence* seq = st->CreateSequence(output);
        *seq = *var0->GetSequence();
        for (const auto& a : *var1->GetSequence()) seq->push_back(a);
    }
}

void GenericIsOp::RunImpl(XCVMState* st) {
    XCVMVar* var0 = st->GetXCVMVar(a);
    XCVMVar* var1 = st->GetXCVMVar(b);

    bool result = false;
    if (var0->kind() != var1->kind()) {
        // We are sure the return value is false.
    } else if (var0->kind() == XCVMVar::Kind::kArray) {
        chainerx::Array a = var0->GetArray();
        chainerx::Array b = var1->GetArray();
        if (a.ndim() == 0 && b.ndim() == 0 &&
            a.dtype() == chainerx::Dtype::kBool &&
            b.dtype() == chainerx::Dtype::kBool) {
            result = (static_cast<bool>(chainerx::AsScalar(a)) ==
                      static_cast<bool>(chainerx::AsScalar(b)));
        } else if (a.dtype() != b.dtype() || a.shape() != b.shape() ||
                   a.raw_data() != b.raw_data()) {
            // We are sure the return value is false.
        } else {
            WARN_ONCE("`is` keyword for non boolean scalars might be wrong");
        }
    } else {
        WARN_ONCE("`is` keyword for sequences might be wrong");
    }
    st->SetArray(output, MakeHostArray(chainerx::Dtype::kBool, {}, &result));
}

void GenericAccumulateGradOp::RunImpl(XCVMState* st) {
    XCVMVar* var0 = st->GetXCVMVar(a);
    XCVMVar* var1 = st->GetXCVMVar(b);
    CHECK(var0->kind() == var1->kind()) << var0->DebugString() << " vs " << var1->DebugString();

    switch (var0->kind()) {
    case XCVMVar::Kind::kArray: {
        st->SetArray(output, var0->GetArray() + var1->GetArray());
        break;
    }
    case XCVMVar::Kind::kSequence: {
        const XCVMSequence& seq0 = *var0->GetSequence();
        const XCVMSequence& seq1 = *var1->GetSequence();
        XCVMSequence* out = st->CreateSequence(output);
        CHECK(seq0.size() == seq1.size()) << var0->DebugString() << " vs " << var1->DebugString();
        out->resize(seq0.size());
        for (size_t i = 0; i < seq0.size(); ++i) {
            if (!seq0[i].IsNull() && !seq1[i].IsNull()) {
                (*out)[i] = XCVMVar(seq0[i].GetArray() + seq1[i].GetArray());
            } else if (!seq0[i].IsNull()) {
                (*out)[i] = seq0[i];
            } else if (!seq1[i].IsNull()) {
                (*out)[i] = seq1[i];
            }
        }
        break;
    }

    case XCVMVar::Kind::kOpaque:
    case XCVMVar::Kind::kNull:
        CHECK(false) << var0->DebugString();
    }
}

}  // namespace runtime
}  // namespace oniku
