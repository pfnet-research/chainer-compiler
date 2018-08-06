#include "node.h"

namespace oniku {

Node::Node(const onnx::NodeProto& xnode,
           const std::vector<Value*>& inputs,
           const std::vector<Value*>& outputs)
    : inputs_(inputs),
      outputs_(outputs),
      name_(xnode.name()),
      op_type_(xnode.op_type()),
      domain_(xnode.domain()),
      doc_string_(xnode.doc_string()) {

}

Node::~Node() {
}

}  // namespace oniku
