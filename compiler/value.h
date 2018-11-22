#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnx/onnx_pb.h>

namespace oniku {

class Node;
class Tensor;
class Type;

class Value {
public:
    enum class Kind {
        kTemp = 0,
        kInput = 1,
        kOutput = 2,
        kNull = 4,
    };

    Value(const onnx::ValueInfoProto& xvalue, Kind kind);
    explicit Value(const std::string& name, Kind kind = Kind::kTemp);
    Value(const std::string& name, const Type& type, Kind kind = Kind::kTemp);
    ~Value();

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    void ToONNX(onnx::ValueInfoProto* xvalue) const;
    std::string DebugString() const;

    bool IsTemp() const {
        return kind_ == Kind::kTemp;
    }
    bool IsInput() const {
        return static_cast<int>(kind_) & static_cast<int>(Kind::kInput);
    }
    bool IsOutput() const {
        return static_cast<int>(kind_) & static_cast<int>(Kind::kOutput);
    }
    bool IsNull() const {
        return static_cast<int>(kind_) & static_cast<int>(Kind::kNull);
    }

    const std::string& name() const {
        return name_;
    }

    const Type& type() const {
        return *type_;
    }
    void set_type(Type* type);
    Type* mutable_type() {
        return type_.get();
    }

    int64_t GetNBytes() const;

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

private:
    Kind kind_{Kind::kTemp};
    std::string name_;
    std::unique_ptr<Type> type_;
    std::string doc_string_;
    std::unique_ptr<Tensor> initializer_;

    std::vector<Node*> users_;
    Node* producer_ = nullptr;
    // This should be used only during gradient calculation.
    Value* grad_ = nullptr;
    int counter_ = 0;
};

std::ostream& operator<<(std::ostream& os, const Value::Kind& kind);

}  // namespace oniku
