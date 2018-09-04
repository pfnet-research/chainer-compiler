#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

#include <compiler/dtype.h>

namespace oniku {

class Type {
public:
    explicit Type(const onnx::TypeProto& xtype);
    Type(Dtype dtype, const std::vector<int64_t>& dims);

    explicit Type(const Type& type);
    Type& operator=(const Type&) = delete;

    void ToONNX(onnx::TypeProto* xtype) const;

    Dtype dtype() const {
        return dtype_;
    }
    void set_dtype(Dtype dtype) {
        dtype_ = dtype;
    }

    const std::vector<int64_t>& dims() const {
        return dims_;
    }

    int64_t NumElements() const;

    bool is_known() const {
        return is_known_;
    }

private:
    Dtype dtype_{Dtype::kUnknown};
    std::vector<int64_t> dims_;
    std::vector<std::string> dim_params_;
    std::vector<std::string> denotations_;
    bool is_known_{true};
};

}  // namespace oniku
