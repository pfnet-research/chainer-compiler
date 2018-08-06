#pragma once

#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

namespace oniku {

class Node;

class Value {
public:
    enum class Kind {
        kInput,
        kOutput,
        kTemp,
    };

    Value(const onnx::ValueInfoProto& xnode, Kind kind);
    ~Value();

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    Kind kind() const { return kind_; }
    const std::string& name() const { return name_; }
    const onnx::TypeProto& type() const { return type_; }
    const std::string& doc_string() const { return doc_string_; }
    const std::vector<Node*>& users() const { return users_; }

private:
    Kind kind_;
    std::string name_;
    // TODO(hamaji): Consider introducing oniku::Type.
    onnx::TypeProto type_;
    std::string doc_string_;

    std::vector<Node*> users_;
};

}  // namespace oniku
