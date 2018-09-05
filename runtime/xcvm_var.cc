#include "xcvm_var.h"

#include <common/log.h>
#include <common/strutil.h>

namespace oniku {
namespace runtime {

XCVMVar::XCVMVar(Kind kind)
    : kind_(kind) {
    CHECK(kind_ != Kind::kArray);
}

XCVMVar::XCVMVar(xchainer::Array array)
    : kind_(Kind::kArray),
      array_(array) {
}

xchainer::Array XCVMVar::GetArray() {
    CHECK(kind_ == Kind::kArray) << static_cast<int>(kind_);
    return *array_;
}

std::vector<xchainer::Array>* XCVMVar::GetSequence() {
    CHECK(kind_ == Kind::kSequence) << static_cast<int>(kind_);
    return &sequence_;
}

std::string XCVMVar::ToString() const {
    switch (kind_) {
    case Kind::kArray:
        return array_->shape().ToString();
    case Kind::kSequence:
        return StrCat(
            '[',
            Join(MapToString(sequence_, [this](const xchainer::Array a) {
                        return a.shape().ToString();
                    })),
            ']');
    }
    CHECK(false);
}

std::string XCVMVar::DebugString() const {
    switch (kind_) {
    case Kind::kArray:
        return array_->ToString();
    case Kind::kSequence:
        return StrCat(
            '[',
            Join(MapToString(sequence_, [this](const xchainer::Array a) {
                        return a.ToString();
                    })),
            ']');
    }
    CHECK(false);
}

}  // namespace runtime
}  // namespace oniku
