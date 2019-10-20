#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <absl/types/variant.h>

#include <chainerx/array.h>

#include <common/log.h>
#include <compiler/dtype.h>
#include <compiler/onnx.h>

namespace chainer_compiler {

class Node;

class Tensor {
public:
    explicit Tensor(const onnx::TensorProto& xtensor);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(const std::string& name, const Tensor& t);
    Tensor(const std::string& name, chainerx::Array ary);

    void ToONNX(onnx::TensorProto* xtensor) const;
    std::string DebugString() const;

    const std::vector<int64_t> dims() const;
    Dtype dtype() const;
    const std::string& name() const {
        return name_;
    }
    const std::string& doc_string() const {
        return doc_string_;
    }

    int ElementSize() const;
    int64_t NumElements() const;

    template <typename T>
    T Get(int index) const {
        CHECK_EQ(dtype().SizeOf(), sizeof(T));
        return static_cast<const T*>(GetRawData())[index];
    }

    const void* GetRawData() const;

    bool IsArray() const;

    const chainerx::Array& chx() const;

    const std::vector<std::string>& str() const;

private:
    // Must be a C-contiguous array.
    const absl::variant<chainerx::Array, std::vector<std::string>> data_;
    std::string name_;
    std::string doc_string_;
};

}  // namespace chainer_compiler
