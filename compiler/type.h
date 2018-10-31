#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnx/onnx_pb.h>

#include <compiler/dtype.h>

namespace oniku {

class Type {
public:
    enum class Kind {
        kTensor, kSequence, kMap, kOpaque
    };

    explicit Type(Kind kind);
    explicit Type(const onnx::TypeProto& xtype);
    Type(Dtype dtype, const std::vector<int64_t>& dims);

    explicit Type(const Type& type);
    Type& operator=(const Type&) = delete;

    void ToONNX(onnx::TypeProto* xtype) const;

    Kind kind() const {
        return kind_;
    }

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

    int64_t GetNBytes() const;

    bool is_known() const {
        return is_known_;
    }

private:
    Kind kind_{Kind::kTensor};
    Dtype dtype_{Dtype::kUnknown};
    std::vector<int64_t> dims_;
    std::vector<std::string> dim_params_;
    std::vector<std::string> denotations_;
    std::unique_ptr<Type> sequence_;
    bool is_known_{true};
};

std::ostream& operator<<(std::ostream& os, const Type::Kind& kind);

}  // namespace oniku
