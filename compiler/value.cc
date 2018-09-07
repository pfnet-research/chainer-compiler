#include "value.h"

#include <algorithm>

#include <common/log.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/type.h>

namespace oniku {

Value::Value(const onnx::ValueInfoProto& xvalue, Kind kind)
    : kind_(kind), name_(xvalue.name()), type_(xvalue.type()), doc_string_(xvalue.doc_string()) {
}

Value::Value(const std::string& name, Kind kind) : kind_(kind), name_(name), type_(Dtype::kUnknown, {}) {
}

Value::Value(const std::string& name, const Type& type, Kind kind) : kind_(kind), name_(name), type_(type) {
}

Value::~Value() {
    CHECK(grad_ == nullptr);
}

void Value::ToONNX(onnx::ValueInfoProto* xvalue) const {
    DUMP_STRING(xvalue, name);
    type_.ToONNX(xvalue->mutable_type());
    DUMP_STRING(xvalue, doc_string);
}

void Value::ResetInitializer(std::unique_ptr<Tensor>&& tensor) {
    initializer_.reset(tensor.release());
}

int64_t Value::GetTotalSize() const {
    return type_.GetTotalSize();
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

}  // namespace oniku
