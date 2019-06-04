#pragma once

#include "common/log.h"

#include <chainerx/float16.h>
#include <chainerx/scalar.h>

namespace chainer_compiler {
namespace runtime {

class StrictScalar {
public:
    StrictScalar() = default;
    StrictScalar(chainerx::Dtype t, chainerx::Scalar d, bool h = false) : data_(d), dtype_(t), host_(h) {
    }

    chainerx::Dtype dtype() const {
        return dtype_;
    }
    bool host() const {
        return host_;
    }

    explicit operator chainerx::Scalar() const {
        return data_;
    }

    explicit operator bool() const {
        return static_cast<bool>(static_cast<chainerx::Scalar>(*this));
    }

    explicit operator int64_t() const {
        return static_cast<int64_t>(static_cast<chainerx::Scalar>(*this));
    }

    explicit operator uint8_t() const {
        return static_cast<uint8_t>(static_cast<chainerx::Scalar>(*this));
    }

private:
    chainerx::Scalar data_;
    chainerx::Dtype dtype_;
    bool host_;
};

}  // namespace runtime
}  // namespace chainer_compiler
