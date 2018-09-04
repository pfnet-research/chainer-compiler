#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

#include <compiler/type.h>

namespace oniku {

class Node;
class Tensor;
class Type;

class Value {
public:
    enum class Kind {
        kInput,
        kOutput,
        kTemp,
        kNull,
    };

    Value(const onnx::ValueInfoProto& xvalue, Kind kind);
    explicit Value(const std::string& name, Kind kind = Kind::kTemp);
    Value(const std::string& name, const Type& type, Kind kind = Kind::kTemp);
    ~Value();

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    void ToONNX(onnx::ValueInfoProto* xvalue) const;

    Kind kind() const {
        return kind_;
    }
    const std::string& name() const {
        return name_;
    }

    const Type& type() const {
        return type_;
    }
    void set_type(const Type& type) {
        type_ = type;
    }
    Type* mutable_type() {
        return &type_;
    }

    const std::string& doc_string() const {
        return doc_string_;
    }

    const Tensor* initializer() const {
        return initializer_.get();
    }
    void ResetInitializer(std::unique_ptr<Tensor>&& tensor);

    const std::vector<Node*>& users() const {
        return users_;
    }
    void AddUser(Node* user);
    void DetachUser(const Node* user);

    Node* producer() const {
        return producer_;
    }
    void SetProducer(Node* producer);

    Value* grad() const {
        return grad_;
    }
    void set_grad(Value* grad) {
        grad_ = grad;
    }

    // Generate a unique ID for other values associated with this object.
    int Counter() {
        return counter_++;
    }

    bool IsNull() const {
        return kind_ == Kind::kNull;
    }

private:
    Kind kind_;
    std::string name_;
    Type type_;
    std::string doc_string_;
    std::unique_ptr<Tensor> initializer_;

    std::vector<Node*> users_;
    Node* producer_ = nullptr;
    // This should be used only during gradient calculation.
    Value* grad_ = nullptr;
    int counter_ = 0;
};

}  // namespace oniku
