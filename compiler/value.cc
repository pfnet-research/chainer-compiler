#include "value.h"

namespace oniku {

Value::Value(const onnx::ValueInfoProto& xvalue, Kind kind)
    : kind_(kind),
      name_(xvalue.name()),
      type_(xvalue.type()),
      doc_string_(xvalue.doc_string()) {
}

Value::~Value() {
}

}  // namespace oniku
