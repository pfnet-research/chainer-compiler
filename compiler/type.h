#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <compiler/onnx.h>

#include <compiler/dtype.h>

namespace chainer_compiler {

class Type {
public:
    enum class Kind { kTensor, kSequence, kMap, kOpaque };

    explicit Type(Kind kind);
    explicit Type(const onnx::TypeProto& xtype);
    Type();
    explicit Type(Dtype dtype);
    Type(Dtype dtype, const std::vector<int64_t>& dims);

    explicit Type(const Type& type);
    Type& operator=(const Type&) = delete;

    void ToONNX(onnx::TypeProto* xtype) const;
    std::string DebugString() const;

    Kind kind() const {
        return kind_;
    }

    Dtype dtype() const {
        return dtype_;
    }
    void set_dtype(Dtype dtype) {
        dtype_ = dtype;
    }

    size_t ndim() const {
        return dims_.size();
    }

    const std::vector<int64_t>& dims() const {
        return dims_;
    }

    int64_t dim(size_t i) const {
        return dims_[i];
    }

    void set_dim_param(size_t i, const std::string& param);

    const std::string& denotation() const {
        return denotation_;
    }
    void set_denotation(const std::string& denotation) {
        denotation_ = denotation;
    }

    int64_t NumElements() const;

    int64_t GetNBytes() const;

    bool HasKnownShape() const;

private:
    Kind kind_{Kind::kTensor};
    Dtype dtype_{Dtype::kUnknown};
    std::vector<int64_t> dims_;
    std::vector<std::string> dim_params_;
    std::vector<std::string> dim_denotations_;
    std::unique_ptr<Type> sequence_;
    std::string denotation_;
    bool has_known_shape_{true};
};

std::ostream& operator<<(std::ostream& os, const Type::Kind& kind);

}  // namespace chainer_compiler
