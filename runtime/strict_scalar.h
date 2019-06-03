#pragma once

#include "common/log.h"

#include <chainerx/scalar.h>
#include <chainerx/float16.h>

namespace chainer_compiler {
namespace runtime {

class StrictScalar {
public:
    typedef union {
        bool bool_;
        int8_t int8_;
        int16_t int16_;
        int32_t int32_;
        int64_t int64_;
        uint8_t uint8_;
        uint16_t float16_;
        float float32_;
        double float64_;
    } InternalType;

    StrictScalar() = default;
    StrictScalar(chainerx::Dtype t, InternalType d, bool h = false) : data_(d), dtype_(t), host_(h) {}

    chainerx::Dtype dtype() const { return dtype_; }
    const InternalType& data() const { return data_; }
    bool host() const { return host_; }

    explicit operator chainerx::Scalar() const {
        switch(dtype_) {
            case chainerx::Dtype::kBool:
                return chainerx::Scalar(data_.bool_);
            case chainerx::Dtype::kInt8:
                return chainerx::Scalar(data_.int8_);
            case chainerx::Dtype::kInt16:
                return chainerx::Scalar(data_.int16_);
            case chainerx::Dtype::kInt32:
                return chainerx::Scalar(data_.int32_);
            case chainerx::Dtype::kInt64:
                return chainerx::Scalar(data_.int64_);
            case chainerx::Dtype::kUInt8:
                return chainerx::Scalar(data_.uint8_);
            case chainerx::Dtype::kFloat16:
                return chainerx::Scalar(chainerx::Float16::FromData(data_.float16_));
            case chainerx::Dtype::kFloat32:
                return chainerx::Scalar(data_.float32_);
            case chainerx::Dtype::kFloat64:
                return chainerx::Scalar(data_.float64_);
            default:
                CHECK(false);
        }
    }

    explicit operator bool() const {
        return static_cast<bool>(static_cast<chainerx::Scalar>(*this));
    }

    explicit operator int64_t() const {
        return static_cast<int64_t>(static_cast<chainerx::Scalar>(*this));
    }

private:
    InternalType data_;
    chainerx::Dtype dtype_;
    bool host_;
};

}  // namespace runtime
}  // namespace chainer_compiler
