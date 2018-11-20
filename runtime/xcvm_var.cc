#include "xcvm_var.h"

#include <common/log.h>
#include <common/strutil.h>

namespace oniku {
namespace runtime {

XCVMVar::XCVMVar() : kind_(Kind::kNull) {
}

XCVMVar::XCVMVar(Kind kind) : kind_(kind) {
    CHECK_NE(kind_, Kind::kArray);
    CHECK_NE(kind_, Kind::kOpaque);
    if (kind_ == Kind::kSequence) {
        sequence_.reset(new XCVMSequence());
    }
}

XCVMVar::XCVMVar(chainerx::Array array)
    : kind_(Kind::kArray), array_(array) {
}

XCVMVar::XCVMVar(XCVMOpaque* opaque)
    : kind_(Kind::kOpaque), opaque_(opaque) {
}

const chainerx::Array& XCVMVar::GetArray() const {
    CHECK_EQ(kind_, Kind::kArray);
    return array_;
}

XCVMSequence* XCVMVar::GetSequence() const {
    CHECK_EQ(kind_, Kind::kSequence);
    return sequence_.get();
}

XCVMOpaque* XCVMVar::GetOpaque() const {
    CHECK_EQ(kind_, Kind::kOpaque);
    return opaque_.get();
}

int64_t XCVMVar::GetTotalSize() const {
    int64_t size = 0;
    switch (kind_) {
        case Kind::kArray:
            size = array_.GetNBytes();
            break;
        case Kind::kSequence:
            for (const XCVMVar& v : *sequence_) size += v.GetArray().GetNBytes();
            break;
        case Kind::kOpaque:
        case Kind::kNull:
            CHECK(false) << DebugString();
    }
    return size;
}

char XCVMVar::Sigil() const {
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

std::string XCVMVar::ToString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_.shape().ToString();
        case Kind::kSequence:
            return StrCat('[', JoinString(MapToString(*sequence_, [this](const XCVMVar& v) { return v.ToString(); })), ']');
        case Kind::kOpaque:
            return opaque_->ToString();
        case Kind::kNull:
            return "(null)";
    }
    CHECK(false);
}

std::string XCVMVar::DebugString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_.ToString();
        case Kind::kSequence:
            return StrCat('[', JoinString(MapToString(*sequence_, [this](const XCVMVar& v) { return v.DebugString(); })), ']');
        case Kind::kOpaque:
            return opaque_->DebugString();
        case Kind::kNull:
            return "(null)";
    }
    CHECK(false);
}

std::vector<chainerx::Array> NonOptional(const XCVMSequence& seq) {
    std::vector<chainerx::Array> r;
    for (const XCVMVar& v : seq) {
        r.push_back(v.GetArray());
    }
    return r;
}

std::ostream& operator<<(std::ostream& os, const XCVMVar::Kind& kind) {
    static const char* kNames[] = { "Array", "Sequence", "Opaque", "Null" };
    int k = static_cast<int>(kind);
    if (k >= 0 && k < sizeof(kNames) / sizeof(kNames[0])) {
        os << kNames[k];
    } else {
        os << "???(" << k << ")";
    }
    return os;
}

}  // namespace runtime
}  // namespace oniku
