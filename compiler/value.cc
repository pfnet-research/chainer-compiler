#include "value.h"

#include <compiler/serializer_util.h>
#include <compiler/tensor.h>

namespace oniku {

Value::Value(const onnx::ValueInfoProto& xvalue, Kind kind)
    : kind_(kind), name_(xvalue.name()), type_(xvalue.type()), doc_string_(xvalue.doc_string()) {}

Value::~Value() {}

void Value::ToONNX(onnx::ValueInfoProto* xvalue) {
    DUMP_STRING(xvalue, name);
    *xvalue->mutable_type() = type_;
    DUMP_STRING(xvalue, doc_string);
}

void Value::ResetInitializer(std::unique_ptr<Tensor>&& tensor) { initializer_.reset(tensor.release()); }

}  // namespace oniku
