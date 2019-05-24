#include "runtime/chxvm_var.h"

#include <common/log.h>
#include <common/strutil.h>

namespace chainer_compiler {
namespace runtime {

const std::vector<chainerx::Array>& ChxVMOpaque::GetArrays() const {
    CHECK(retained_arrays_) << "SetRetainedArrays was not called: " << DebugString();
    return *retained_arrays_;
}

void ChxVMOpaque::SetRetainedArrays(const std::vector<chainerx::Array>& retained_arrays) {
    retained_arrays_.reset(new std::vector<chainerx::Array>(retained_arrays));
}

ChxVMVar::ChxVMVar() : kind_(Kind::kNull) {
}

ChxVMVar::ChxVMVar(Kind kind) : kind_(kind) {
    CHECK_NE(kind_, Kind::kArray);
    CHECK_NE(kind_, Kind::kOpaque);
    if (kind_ == Kind::kSequence) {
        sequence_.reset(new ChxVMSequence());
    }
}

ChxVMVar::ChxVMVar(chainerx::Array array) : kind_(Kind::kArray), array_(array) {
}

ChxVMVar::ChxVMVar(ChxVMOpaque* opaque) : kind_(Kind::kOpaque), opaque_(opaque) {
}

const chainerx::Array& ChxVMVar::GetArray() const {
    CHECK_EQ(kind_, Kind::kArray);
    return array_;
}

ChxVMSequence* ChxVMVar::GetSequence() const {
    CHECK_EQ(kind_, Kind::kSequence);
    return sequence_.get();
}

ChxVMOpaque* ChxVMVar::GetOpaque() const {
    CHECK_EQ(kind_, Kind::kOpaque);
    return opaque_.get();
}

int64_t ChxVMVar::GetNBytes() const {
    int64_t size = 0;
    switch (kind_) {
        case Kind::kArray:
            size = array_.GetNBytes();
            break;
        case Kind::kSequence:
            for (const ChxVMVar& v : *sequence_) size += v.GetArray().GetNBytes();
            break;
        case Kind::kOpaque:
        case Kind::kNull:
            CHECK(false) << DebugString();
    }
    return size;
}

std::vector<chainerx::Array> ChxVMVar::GetArrays() const {
    switch (kind_) {
        case Kind::kArray:
            return {array_};
        case Kind::kSequence: {
            std::vector<chainerx::Array> arrays;
            for (const ChxVMVar& v : *sequence_) {
                for (const chainerx::Array a : v.GetArrays()) {
                    arrays.push_back(a);
                }
            }
            return arrays;
        }
        case Kind::kOpaque:
            return opaque_->GetArrays();

        case Kind::kNull:
            return {};

        default:
            CHECK(false) << DebugString();
            return {};
    }
}

char ChxVMVar::Sigil() const {
    switch (kind_) {
        case Kind::kArray:
            return '@';
        case Kind::kSequence:
            return '$';
        case Kind::kOpaque:
            return '*';
        case Kind::kNull:
            CHECK(false);
    }
    CHECK(false);
}

std::string ChxVMVar::ToString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_.shape().ToString();
        case Kind::kSequence:
            return StrCat('[', JoinString(MapToString(*sequence_, [this](const ChxVMVar& v) { return v.ToString(); })), ']');
        case Kind::kOpaque:
            return opaque_->ToString();
        case Kind::kNull:
            return "(null)";
    }
    CHECK(false);
}

std::string ChxVMVar::DebugString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_.ToString();
        case Kind::kSequence:
            return StrCat('[', JoinString(MapToString(*sequence_, [this](const ChxVMVar& v) { return v.DebugString(); })), ']');
        case Kind::kOpaque:
            return opaque_->DebugString();
        case Kind::kNull:
            return "(null)";
    }
    CHECK(false);
}

std::vector<chainerx::Array> NonOptional(const ChxVMSequence& seq) {
    std::vector<chainerx::Array> r;
    for (const ChxVMVar& v : seq) {
        r.push_back(v.GetArray());
    }
    return r;
}

std::ostream& operator<<(std::ostream& os, const ChxVMVar::Kind& kind) {
    static const char* kNames[] = {"Array", "Sequence", "Opaque", "Null"};
    int k = static_cast<int>(kind);
    if (k >= 0 && k < sizeof(kNames) / sizeof(kNames[0])) {
        os << kNames[k];
    } else {
        os << "???(" << k << ")";
    }
    return os;
}

}  // namespace runtime
}  // namespace chainer_compiler
