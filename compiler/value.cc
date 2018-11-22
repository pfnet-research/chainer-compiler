#include "value.h"

#include <algorithm>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/type.h>

namespace oniku {

Value::Value(const onnx::ValueInfoProto& xvalue, Kind kind)
    : kind_(kind), name_(xvalue.name()), type_(new Type(xvalue.type())), doc_string_(xvalue.doc_string()) {
}

Value::Value(const std::string& name, Kind kind) : kind_(kind), name_(name), type_(new Type()) {
}

Value::Value(const std::string& name, const Type& type, Kind kind) : kind_(kind), name_(name), type_(new Type(type)) {
}

Value::~Value() {
    CHECK(grad_ == nullptr);
}

void Value::ToONNX(onnx::ValueInfoProto* xvalue) const {
    DUMP_STRING(xvalue, name);
    type_->ToONNX(xvalue->mutable_type());
    DUMP_STRING(xvalue, doc_string);
}

std::string Value::DebugString() const {
    onnx::ValueInfoProto xvalue;
    ToONNX(&xvalue);
    return xvalue.DebugString();
}

void Value::ResetInitializer(std::unique_ptr<Tensor>&& tensor) {
    initializer_.reset(tensor.release());
}

void Value::set_type(Type* type) {
    type_.reset(type);
}

int64_t Value::GetNBytes() const {
    return type_->GetNBytes();
}

void Value::AddUser(Node* user) {
    users_.push_back(user);
}

void Value::DetachUser(const Node* user) {
    auto found = std::find(users_.begin(), users_.end(), user);
    CHECK(found != users_.end());
    users_.erase(found);
}

void Value::SetProducer(Node* producer) {
    producer_ = producer;
}

std::ostream& operator<<(std::ostream& os, const Value::Kind& kind) {
    if (kind == Value::Kind::kTemp) {
        return os << "Temp";
    }
    int k = static_cast<int>(kind);
    std::vector<std::string> out;
    if (k & static_cast<int>(Value::Kind::kInput)) out.push_back("Input");
    if (k & static_cast<int>(Value::Kind::kOutput)) out.push_back("Output");
    if (k & static_cast<int>(Value::Kind::kNull)) out.push_back("Null");
    if (out.empty()) {
        os << "???(" << k << ")";
    } else {
        os << JoinString(out, "|");
    }
    return os;
}

}  // namespace oniku
