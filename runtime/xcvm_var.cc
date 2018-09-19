#include "xcvm_var.h"

#include <common/log.h>
#include <common/strutil.h>

namespace oniku {
namespace runtime {

XCVMVar::XCVMVar(Kind kind) : kind_(kind) {
    CHECK(kind_ != Kind::kArray);
}

XCVMVar::XCVMVar(chainerx::Array array) : kind_(Kind::kArray), array_(array) {
}

const chainerx::Array& XCVMVar::GetArray() {
    CHECK(kind_ == Kind::kArray) << static_cast<int>(kind_);
    return *array_;
}

std::vector<chainerx::Array>* XCVMVar::GetSequence() {
    CHECK(kind_ == Kind::kSequence) << static_cast<int>(kind_);
    return &sequence_;
}

int64_t XCVMVar::GetTotalSize() const {
    int64_t size = 0;
    switch (kind_) {
        case Kind::kArray:
            size = array_->GetNBytes();
            break;
        case Kind::kSequence:
            for (chainerx::Array a : sequence_) size += a.GetNBytes();
            break;
    }
    return size;
}

char XCVMVar::Sigil() const {
    switch (kind_) {
        case Kind::kArray:
            return '@';
        case Kind::kSequence:
            return '$';
    }
    CHECK(false);
}

std::string XCVMVar::ToString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_->shape().ToString();
        case Kind::kSequence:
            return StrCat('[', Join(MapToString(sequence_, [this](const chainerx::Array a) { return a.shape().ToString(); })), ']');
    }
    CHECK(false);
}

std::string XCVMVar::DebugString() const {
    switch (kind_) {
        case Kind::kArray:
            return array_->ToString();
        case Kind::kSequence:
            return StrCat('[', Join(MapToString(sequence_, [this](const chainerx::Array a) { return a.ToString(); })), ']');
    }
    CHECK(false);
}

}  // namespace runtime
}  // namespace oniku
