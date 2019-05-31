#include "runtime/chainerx_util.h"
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
        val_ = std::make_shared<ChxVMSequence>();
    }
}

ChxVMVar::ChxVMVar(chainerx::Array array) : kind_(Kind::kArray), val_(array) {
}

ChxVMVar::ChxVMVar(ChxVMOpaque* opaque) : kind_(Kind::kOpaque), val_(std::shared_ptr<ChxVMOpaque>(opaque)) {
}

ChxVMVar::ChxVMVar(chainerx::Scalar scalar) : kind_(Kind::kScalar), val_(scalar) {
}

ChxVMVar::ChxVMVar(chainerx::Shape shape) : kind_(Kind::kShape), val_(shape) {
}

const chainerx::Array& ChxVMVar::GetArray() const {
    if (kind_ == Kind::kShape) {
        kind_ = Kind::kArray;
        val_ = runtime::ShapeToArray(absl::get<chainerx::Shape>(val_));
    }
    CHECK_EQ(kind_, Kind::kArray);
    return absl::get<chainerx::Array>(val_);
}

ChxVMSequence* ChxVMVar::GetSequence() const {
    CHECK_EQ(kind_, Kind::kSequence);
    return absl::get<std::shared_ptr<ChxVMSequence>>(val_).get();
}

ChxVMOpaque* ChxVMVar::GetOpaque() const {
    CHECK_EQ(kind_, Kind::kOpaque);
    return absl::get<std::shared_ptr<ChxVMOpaque>>(val_).get();
}

const chainerx::Scalar& ChxVMVar::GetScalar() const {
    CHECK_EQ(kind_, Kind::kScalar);
    return absl::get<chainerx::Scalar>(val_);
}

const chainerx::Shape& ChxVMVar::GetShape() const {
    if (kind_ == Kind::kArray) {
        kind_ = Kind::kShape;
        val_ = runtime::ArrayToShape(absl::get<chainerx::Array>(val_));
    }
    CHECK_EQ(kind_, Kind::kShape);
    return absl::get<chainerx::Shape>(val_);
}

int64_t ChxVMVar::GetNBytes() const {
    int64_t size = 0;
    switch (kind_) {
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
    switch (kind_) {
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
        case Kind::kShape:
        case Kind::kScalar:
        case Kind::kNull:
            CHECK(false);
    }
    CHECK(false);
}

std::string ChxVMVar::ToString() const {
    switch (kind_) {
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
    switch (kind_) {
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
