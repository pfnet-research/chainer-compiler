#include "xcvm_var.h"

#include <common/log.h>
#include <common/strutil.h>

namespace oniku {
namespace runtime {

XCVMVar::XCVMVar() : kind_(Kind::kNull) {
}

XCVMVar::XCVMVar(Kind kind) : kind_(kind) {
    CHECK(kind_ != Kind::kArray);
    CHECK(kind_ != Kind::kOpaque);
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
    CHECK(kind_ == Kind::kArray) << static_cast<int>(kind_);
    return array_;
}

XCVMSequence* XCVMVar::GetSequence() const {
    CHECK(kind_ == Kind::kSequence) << static_cast<int>(kind_);
    return sequence_.get();
}

XCVMOpaque* XCVMVar::GetOpaque() const {
    CHECK(kind_ == Kind::kOpaque) << static_cast<int>(kind_);
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
            return StrCat('[', Join(MapToString(*sequence_, [this](const XCVMVar& v) { return v.ToString(); })), ']');
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
            return StrCat('[', Join(MapToString(*sequence_, [this](const XCVMVar& v) { return v.DebugString(); })), ']');
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

}  // namespace runtime
}  // namespace oniku
