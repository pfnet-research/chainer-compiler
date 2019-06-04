#include "runtime/chxvm_var.h"

#include <common/log.h>
#include <common/strutil.h>

#include <runtime/chainerx_util.h>

namespace chainer_compiler {
namespace runtime {

const std::vector<chainerx::Array>& ChxVMOpaque::GetArrays() const {
    CHECK(retained_arrays_) << "SetRetainedArrays was not called: " << DebugString();
    return *retained_arrays_;
}

void ChxVMOpaque::SetRetainedArrays(const std::vector<chainerx::Array>& retained_arrays) {
    retained_arrays_.reset(new std::vector<chainerx::Array>(retained_arrays));
}

ChxVMVar::ChxVMVar() : val_(NullType{}) {
}

ChxVMVar::ChxVMVar(chainerx::Array array) : val_(array) {
}

ChxVMVar::ChxVMVar(ChxVMOpaque* opaque) : val_(std::shared_ptr<ChxVMOpaque>(opaque)) {
}

ChxVMVar::ChxVMVar(chainerx::Scalar scalar) : val_(scalar) {
}

ChxVMVar::ChxVMVar(chainerx::Shape shape) : val_(shape) {
}

ChxVMVar::ChxVMVar(std::shared_ptr<ChxVMSequence> seq) : val_(seq) {
}

const chainerx::Array& ChxVMVar::GetArray() const {
    if (kind() == Kind::kShape) {
        val_ = runtime::ShapeToArray(absl::get<chainerx::Shape>(val_));
    }
    return absl::get<chainerx::Array>(val_);
}

ChxVMSequence* ChxVMVar::GetSequence() const {
    return absl::get<std::shared_ptr<ChxVMSequence>>(val_).get();
}

ChxVMOpaque* ChxVMVar::GetOpaque() const {
    return absl::get<std::shared_ptr<ChxVMOpaque>>(val_).get();
}

const chainerx::Scalar& ChxVMVar::GetScalar() const {
    return absl::get<chainerx::Scalar>(val_);
}

const chainerx::Shape& ChxVMVar::GetShape() const {
    if (kind() == Kind::kArray) {
        val_ = runtime::ArrayToShape(absl::get<chainerx::Array>(val_));
    }
    return absl::get<chainerx::Shape>(val_);
}

int64_t ChxVMVar::GetNBytes() const {
    int64_t size = 0;
    switch (kind()) {
        case Kind::kArray:
            size = absl::get<chainerx::Array>(val_).GetNBytes();
            break;
        case Kind::kSequence:
            for (const ChxVMVar& v : *GetSequence()) size += v.GetArray().GetNBytes();
            break;
        case Kind::kShape:
        case Kind::kScalar:
        case Kind::kOpaque:
        case Kind::kNull:
            CHECK(false) << DebugString();
    }
    return size;
}

std::vector<chainerx::Array> ChxVMVar::GetArrays() const {
    switch (kind()) {
        case Kind::kArray:
            return {GetArray()};
        case Kind::kSequence: {
            std::vector<chainerx::Array> arrays;
            for (const ChxVMVar& v : *GetSequence()) {
                for (const chainerx::Array a : v.GetArrays()) {
                    arrays.push_back(a);
                }
            }
            return arrays;
        }
        case Kind::kOpaque:
            return GetOpaque()->GetArrays();

        case Kind::kNull:
        case Kind::kScalar:
        case Kind::kShape:
            return {};

        default:
            CHECK(false) << DebugString();
            return {};
    }
}

char ChxVMVar::Sigil() const {
    switch (kind()) {
        case Kind::kArray:
            return '$';
        case Kind::kSequence:
            return '@';
        case Kind::kOpaque:
            return '*';
        case Kind::kShape:
        case Kind::kScalar:
        case Kind::kNull:
            CHECK(false);
    }
    CHECK(false);
}

std::string ChxVMVar::ToString() const {
    switch (kind()) {
        case Kind::kArray:
            return absl::get<chainerx::Array>(val_).shape().ToString();
        case Kind::kSequence:
            return StrCat('[', JoinString(MapToString(*GetSequence(), [this](const ChxVMVar& v) { return v.ToString(); })), ']');
        case Kind::kOpaque:
            return GetOpaque()->ToString();
        case Kind::kNull:
            return "(null)";
        case Kind::kShape: {
            std::ostringstream oss;
            oss << "(" << absl::get<chainerx::Shape>(val_).size() << ")";
            return oss.str();
        }
        case Kind::kScalar:
            break;
    }
    CHECK(false);
}

std::string ChxVMVar::DebugString() const {
    switch (kind()) {
        case Kind::kArray:
            return absl::get<chainerx::Array>(val_).ToString();
        case Kind::kSequence:
            return StrCat('[', JoinString(MapToString(*GetSequence(), [this](const ChxVMVar& v) { return v.DebugString(); })), ']');
        case Kind::kOpaque:
            return GetOpaque()->DebugString();
        case Kind::kNull:
            return "(null)";
        case Kind::kShape:
            return absl::get<chainerx::Shape>(val_).ToString();
        case Kind::kScalar:
            break;
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
