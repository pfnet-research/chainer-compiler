#include "xcvm_var.h"

#include <common/log.h>
#include <common/strutil.h>

namespace oniku {
namespace runtime {

XCVMVar::XCVMVar(Kind kind) : kind_(kind) {
    CHECK(kind_ != Kind::kArray);
}

XCVMVar::XCVMVar(chainerx::Array array)
    : kind_(Kind::kArray), array_(array) {
}

XCVMVar::XCVMVar(XCVMOpaque* opaque)
    : kind_(Kind::kOpaque), opaque_(opaque) {
}

const chainerx::Array& XCVMVar::GetArray() {
    CHECK(kind_ == Kind::kArray) << static_cast<int>(kind_);
    return *array_;
}

XCVMSequence* XCVMVar::GetSequence() {
    CHECK(kind_ == Kind::kSequence) << static_cast<int>(kind_);
    return &sequence_;
}

XCVMOpaque* XCVMVar::GetOpaque() {
    CHECK(kind_ == Kind::kOpaque) << static_cast<int>(kind_);
    return opaque_.get();
}

int64_t XCVMVar::GetTotalSize() const {
    int64_t size = 0;
    switch (kind_) {
        case Kind::kArray:
            size = array_->GetNBytes();
            break;
        case Kind::kSequence:
            for (const nonstd::optional<chainerx::Array>& a : sequence_) size += a->GetNBytes();
            break;
        case Kind::kOpaque:
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
    }
    CHECK(false);
}

std::string XCVMVar::ToString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_->shape().ToString();
        case Kind::kSequence:
            return StrCat('[', Join(MapToString(sequence_, [this](const nonstd::optional<chainerx::Array>& a) { return a.has_value() ? a->shape().ToString() : "(null)"; })), ']');
        case Kind::kOpaque:
            return opaque_->ToString();
    }
    CHECK(false);
}

std::string XCVMVar::DebugString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_->ToString();
        case Kind::kSequence:
            return StrCat('[', Join(MapToString(sequence_, [this](const nonstd::optional<chainerx::Array>& a) { return a.has_value() ? a->ToString() : "(null)"; })), ']');
        case Kind::kOpaque:
            return opaque_->DebugString();
    }
    CHECK(false);
}

}  // namespace runtime
}  // namespace oniku
