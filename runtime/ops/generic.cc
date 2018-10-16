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
    }
    CHECK(false);
}

int64_t GetOptionalInt(XCVMState* st, int index, int64_t default_value) {
    nonstd::optional<chainerx::Array> var = st->GetVarOptional(index);
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
            st->SetVar(y, var->GetArray());
            break;

        case XCVMVar::Kind::kSequence:
            const std::vector<chainerx::Array>& s = *var->GetSequence();
            std::vector<chainerx::Array>* d = st->CreateSequence(y);
            CHECK(d->empty());
            *d = s;
            break;
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
    st->SetVar(len, MakeHostArray(chainerx::Dtype::kInt64, {}, &size));
}

void GenericGetItemOp::RunImpl(XCVMState* st) {
    XCVMVar* var = st->GetXCVMVar(v);
    int64_t size = GetSize(var);
    int64_t i = static_cast<int64_t>(chainerx::AsScalar(st->GetVar(index)));
    if (i < 0) i += size;
    switch (var->kind()) {
        case XCVMVar::Kind::kArray:
            CHECK_LT(i, var->GetArray().shape()[0]);
            st->SetVar(output, var->GetArray().At({i}));
            break;

        case XCVMVar::Kind::kSequence:
            const std::vector<chainerx::Array>& v = *var->GetSequence();
            CHECK_LT(i, v.size());
            st->SetVar(output, v[i]);
            break;
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
            st->SetVar(output, var->GetArray().At({chainerx::Slice(start, end, step)}));
            break;
        }

        case XCVMVar::Kind::kSequence: {
            const std::vector<chainerx::Array>& v = *var->GetSequence();
            std::vector<chainerx::Array>* seq = st->CreateSequence(output);
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
            return Stack(*v->GetSequence(), 0);
        };
        st->SetVar(output, to_a(var0) + to_a(var1));
    } else {
        std::vector<chainerx::Array>* seq = st->CreateSequence(output);
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
    st->SetVar(output, MakeHostArray(chainerx::Dtype::kBool, {}, &result));
}

}  // namespace runtime
}  // namespace oniku
