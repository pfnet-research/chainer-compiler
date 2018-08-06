#pragma once

#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

namespace oniku {

class Node;
class Tensor;

class Value {
public:
    enum class Kind {
        kInput,
        kOutput,
        kTemp,
    };

    Value(const onnx::ValueInfoProto& xvalue, Kind kind);
    ~Value();

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    void ToONNX(onnx::ValueInfoProto* xvalue);

    Kind kind() const { return kind_; }
    const std::string& name() const { return name_; }
    const onnx::TypeProto& type() const { return type_; }
    const std::string& doc_string() const { return doc_string_; }

    Tensor* initializer() { return initializer_.get(); }
    void ResetInitializer(std::unique_ptr<Tensor>&& tensor);

private:
    Kind kind_;
    std::string name_;
    // TODO(hamaji): Consider introducing oniku::Type.
    onnx::TypeProto type_;
    std::string doc_string_;
    std::unique_ptr<Tensor> initializer_;
};

}  // namespace oniku
