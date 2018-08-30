#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <compiler/dtype.h>

namespace oniku {

class Node;

class Tensor {
public:
    typedef std::unique_ptr<void, decltype(&std::free)> UniqueData;

    explicit Tensor(const onnx::TensorProto& xtensor);
    ~Tensor();

    // Undefined reference indicates the type is not supported yet.
    template <class T>
    Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, const std::vector<T>& data);
    template <class T>
    Tensor(const std::string& name, Dtype dtype, const std::vector<int64_t>& dims, const std::initializer_list<T>& data)
        : Tensor(name, dtype, dims, std::vector<T>{data}) {
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(const std::string& name, const Tensor& t);

    void ToONNX(onnx::TensorProto* xtensor) const;

    const std::vector<int64_t> dims() const {
        return dims_;
    }
    Dtype dtype() const {
        return dtype_;
    }
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
        CHECK_EQ(dtype_.SizeOf(), sizeof(T));
        return static_cast<T*>(data_.get())[index];
    }

    const void* GetRawData() const {
        return data_.get();
    }

private:
    std::vector<int64_t> dims_;
    Dtype dtype_;
    UniqueData data_;
    std::string name_;
    std::string doc_string_;
};

}  // namespace oniku
