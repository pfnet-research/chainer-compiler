#pragma once

#include <chainerx/scalar.h>

namespace chainer_compiler {
namespace runtime {

class StrictScalar {
public:
    StrictScalar() = default;
    StrictScalar(chainerx::Dtype t, chainerx::Scalar d, bool h) : data_(d), dtype_(t), host_(h) {
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

    template <class T>
    explicit operator T() const {
        return static_cast<T>(static_cast<chainerx::Scalar>(*this));
    }

private:
    chainerx::Scalar data_;
    chainerx::Dtype dtype_;
    bool host_;
};

}  // namespace runtime
}  // namespace chainer_compiler
