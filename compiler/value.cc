#include "value.h"

#include <common/log.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/type.h>

namespace oniku {

Value::Value(const onnx::ValueInfoProto& xvalue, Kind kind)
    : kind_(kind), name_(xvalue.name()), type_(new Type(xvalue.type())), doc_string_(xvalue.doc_string()) {
}

Value::Value(const std::string& name, Kind kind) : kind_(kind), name_(name), type_(new Type(Dtype::kFloat32, {})) {
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

void Value::ResetInitializer(std::unique_ptr<Tensor>&& tensor) {
    initializer_.reset(tensor.release());
}

void Value::AddUser(Node* user) {
    users_.push_back(user);
}

void Value::SetProducer(Node* producer) {
    producer_ = producer;
}

}  // namespace oniku
