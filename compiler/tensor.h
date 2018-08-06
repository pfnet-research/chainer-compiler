#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

namespace oniku {

class Node;

class Tensor {
public:
    typedef std::unique_ptr<void, decltype(&std::free)> UniqueData;

    enum class Dtype {
        kBool = 1,
        kInt8,
        kInt16,
        kInt32,
        kInt64,
        kUInt8,
        kFloat32,
        kFloat64,
    };

    explicit Tensor(const onnx::TensorProto& xtensor);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void ToONNX(onnx::TensorProto* xtensor);

    const std::vector<int64_t> dims() const { return dims_; }
    Dtype dtype() const { return dtype_; }
    const std::string& name() const { return name_; }
    const std::string& doc_string() const { return doc_string_; }

    int64_t NumElements() const;

private:
    std::vector<int64_t> dims_;
    Dtype dtype_;
    UniqueData data_;
    std::string name_;
    std::string doc_string_;
};

}  // namespace oniku
