#include "compiler/node.h"

#include <algorithm>

#include <common/iterator.h>
#include <common/log.h>
#include <common/strutil.h>
#include <compiler/dtype.h>
#include <compiler/graph.h>
#include <compiler/log.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/value.h>
#include <onnx/defs/schema.h>

namespace chainer_compiler {

Node::Node(
        const OpsetList& opsets,
        const onnx::NodeProto& xnode,
        const std::vector<Value*>& inputs,
        const std::vector<Value*>& outputs,
        const std::string& name)
    : NodeBase(opsets, xnode, inputs, outputs),
      inputs_(inputs),
      outputs_(outputs),
      name_(name.empty() ? xnode.name() : name),
      domain_(xnode.domain()),
      doc_string_(xnode.doc_string()) {
    {
        std::vector<std::string> domains;
        std::vector<int64_t> versions;
        for (const auto i : opsets) {
            domains.push_back(i.domain());
            versions.push_back(i.version());
        }
        set_chainer_onnx_domain(domains);
        set_chainer_onnx_version(versions);
    }

    if (domain_ == onnx::ONNX_DOMAIN && HasPrefix(xnode.op_type(), "Chainer")) {
        domain_ = CHAINER_ONNX_DOMAIN;
    }
    // TODO(take-cheeze): Handle Resize-11
    if (op_type_ == Node::kResize && inputs_.size() == 2) {
        inputs_.push_back(inputs_[1]);
    }
    Validate();
    if (name_.empty() && !outputs.empty()) {
        name_ = output(0)->name();
    }
}

Node::Node(
        const std::string& name,
        OpType op_type,
        const std::vector<Value*>& inputs,
        const std::vector<Value*>& outputs,
        const std::string& domain,
        const OpsetList& opsets)
    : NodeBase(op_type), inputs_(inputs), outputs_(outputs), name_(name), domain_(domain) {
    {
        std::vector<std::string> domains;
        std::vector<int64_t> versions;
        for (const auto i : opsets) {
            domains.push_back(i.domain());
            versions.push_back(i.version());
        }
        set_chainer_onnx_domain(domains);
        set_chainer_onnx_version(versions);
    }
    ValidateNumInputsOutputs(inputs, outputs);
    SetDefaultAttributeValues();
}

Node::~Node() {
}

void Node::ToONNX(onnx::NodeProto* xnode, const OpsetList& opsets, bool validate) const {
    if (validate) {
        CHECK(ValidateWithSchema(opsets));
    }

    for (const auto& value : inputs_) {
        xnode->add_input(value->name());
    }
    for (const auto& value : outputs_) {
        xnode->add_output(value->name());
    }

    DUMP_STRING(xnode, name);
    xnode->set_op_type(OpTypeToString(op_type_));
    DUMP_STRING(xnode, domain);
    DUMP_STRING(xnode, doc_string);

    const OpsetList node_opsets = OpsetImports();
    bool ignore_opset_imports = node_opsets.size() == opsets.size();
    if (ignore_opset_imports) {
        for (size_t i = 0; i < node_opsets.size(); ++i) {
            if (node_opsets[i].domain() != opsets[i].domain() || node_opsets[i].version() != opsets[i].version()) {
                ignore_opset_imports = false;
            }
        }
    }

    FillONNXAttributes(xnode, ignore_opset_imports);
}

std::string Node::DebugString() const {
    onnx::NodeProto xnode;
    ToONNX(&xnode, OpsetImports(), false);
    return xnode.DebugString();
}

void Node::Validate() const {
    ValidateNumInputsOutputs(inputs_, outputs_);
    ValidateAttributes();

    if (op_type_ == Node::kLoop) {
        CHECK(body_.get()) << "Loop without body:\n" << DebugString();
        int num_loop_inputs = inputs_.size();
        int num_loop_outputs = outputs_.size();
        int num_body_inputs = body_->input_values().size();
        int num_body_outputs = body_->output_values().size();
        int num_states = num_loop_inputs - 2;
        int num_scans = num_body_outputs - 1 - num_states;
        CHECK_LT(2, num_loop_inputs) << "Loop should have at least 3 inputs:\n" << DebugString();
        CHECK_LT(0, num_loop_outputs) << "Loop should have at least 1 outputs:\n" << DebugString();
        CHECK_EQ(num_body_inputs, num_states + 2) << "Inconsistent numbers of inputs for Loop:\n" << DebugString();
        CHECK_EQ(num_loop_outputs, num_states + num_scans) << "Inconsistent numbers of outputs for Loop:\n" << DebugString();
        Value* max_trip_count = input(0);
        Value* terminal_condition = input(1);
        CHECK(!max_trip_count->IsNull() || !terminal_condition->IsNull()) << "Inifinite Loop:\n" << DebugString();

    } else if (op_type_ == Node::kIf) {
        CHECK_LT(0, inputs_.size()) << "If should have at least 1 inputs:\n" << DebugString();
        CHECK_EQ(inputs_.size(), then_branch_->input_values().size() + 1) << "Inconsistent number of inputs for If:\n" << DebugString();
        CHECK_EQ(inputs_.size(), else_branch_->input_values().size() + 1) << "Inconsistent number of inputs for If:\n" << DebugString();
        CHECK_EQ(outputs_.size(), then_branch_->output_values().size()) << "Inconsistent number of outputs for If:\n" << DebugString();
        CHECK_EQ(outputs_.size(), else_branch_->output_values().size()) << "Inconsistent number of outputs for If:\n" << DebugString();
    } else if (op_type_ == Node::kChainerGetItem || op_type_ == Node::kChainerGetItemGrad || op_type_ == Node::kChainerSetItem) {
        CHECK_LT(0, inputs_.size()) << op_type_ << " should have at least 1 inputs:\n" << DebugString();
        CHECK_LT(0, slice_specs().size()) << "ChainerGetItem should have at least 1 slice_specs:\n" << DebugString();
        std::vector<bool> must_be_tensor = {true};
        if (op_type_ == Node::kChainerGetItemGrad) {
            must_be_tensor.push_back(true);
        }
        for (int slice_spec : slice_specs()) {
            CHECK_LE(0, slice_spec) << "Negative slice_specs for ChainerGetItem:\n" << DebugString();
            CHECK_GE(4, slice_spec) << "Wrong slice_specs for ChainerGetItem:\n" << DebugString();
            if (slice_spec == 4) {
                must_be_tensor.push_back(false);
            } else {
                for (int i = 0; i < slice_spec; ++i) {
                    must_be_tensor.push_back(true);
                }
            }
        }
        if (op_type_ == Node::kChainerSetItem) {
            must_be_tensor.push_back(true);
        }

        CHECK_EQ(must_be_tensor.size(), inputs_.size()) << "Wrong number of inputs for " << op_type_ << ":\n" << DebugString();
        for (size_t i = 0; i < inputs_.size(); ++i) {
            const Value* value = inputs_[i];
            if (must_be_tensor[i]) {
                CHECK_EQ(Type::Kind::kTensor, value->type().kind()) << i << "th input must be a tensor:\n" << DebugString();
            } else {
                CHECK_EQ(Type::Kind::kSequence, value->type().kind()) << i << "th input must be a sequence:\n" << DebugString();
            }
        }
    }
    // TODO(hamaji): Add more custom validations for other ops.
}

Value* Node::input(int index) const {
    CHECK_LT(index, inputs_.size());
    return inputs_[index];
}

void Node::AddInput(Value* value) {
    inputs_.push_back(value);
    value->AddUser(this);
}

Value* Node::output(int index) const {
    CHECK_LT(index, outputs_.size());
    return outputs_[index];
}

void Node::AddOutput(Value* value, size_t index) {
    if (index == static_cast<size_t>(-1)) {
        outputs_.push_back(value);
    } else {
        CHECK_LE(index, outputs_.size()) << "index=" << index << "\n" << DebugString();
        outputs_.insert(outputs_.begin() + index, value);
    }
    CHECK(!value->producer());
    value->SetProducer(this);
}

void Node::Detach() {
    for (Value* input : inputs_) {
        input->DetachUser(this);
    }
    inputs_.clear();
    outputs_.clear();
    detached_ = true;
}

int Node::GetNumActualInputs() const {
    int count = 0;
    for (const Value* input : inputs_) {
        if (!input->IsNull()) count++;
    }
    return count;
}

void Node::ReplaceInput(Value* f, Value* t) {
    auto found = std::find(inputs_.begin(), inputs_.end(), f);
    CHECK(found != inputs_.end());
    *found = t;
    f->DetachUser(this);
    t->AddUser(this);
}

void Node::ReplaceOutput(Value* f, Value* t) {
    auto found = std::find(outputs_.begin(), outputs_.end(), f);
    CHECK(found != outputs_.end());
    *found = t;
    f->SetProducer(nullptr);
    t->SetProducer(this);
}

std::vector<Graph*> Node::GetSubGraphs() const {
    std::vector<Graph*> subgraphs;
    if (body().get()) subgraphs.push_back(body().get());
    if (then_branch().get()) subgraphs.push_back(then_branch().get());
    if (else_branch().get()) subgraphs.push_back(else_branch().get());
    if (subgraph().get()) subgraphs.push_back(subgraph().get());
    return subgraphs;
}

bool Node::IsGradNode() const {
    // TODO(hamaji): Implement this in a better way.
    return name_.find("Grad") != std::string::npos;
}

bool Node::IsZeroCost() const {
    Node::OpType op = op_type();
    return op == Node::kIdentity || op == Node::kConstant || op == Node::kReshape;
}

std::string Node::ToString() const {
    std::ostringstream oss;
    oss << op_type();
    oss << "(" << JoinString(MapToString(inputs(), [](const Value* v) { return v->name(); })) << ")";
    oss << " -> (" << JoinString(MapToString(outputs(), [](const Value* v) { return v->name(); })) << ")";
    return oss.str();
}

OpsetList Node::OpsetImports() const {
    OpsetList ret;
    CHECK_EQ(chainer_onnx_version().size(), chainer_onnx_domain().size());
    for (size_t i = 0; i < chainer_onnx_version().size(); ++i) {
        onnx::OperatorSetIdProto p;
        p.set_domain(chainer_onnx_domain()[i]);
        p.set_version(chainer_onnx_version()[i]);
        ret.push_back(p);
    }
    CHECK_EQ(chainer_onnx_version().size(), ret.size());
    return ret;
}

int Node::OpVersion() const {
    return GetOpsetVersion(OpsetImports(), domain_);
}

bool Node::ValidateWithSchema(const OpsetList& opsets_, std::string* message) const {
    // TODO(take-cheeze): Fix this workaround
    if (op_type() == Node::kIf) {
        return true;
    }

    const OpsetList& opset = opsets_.empty() ? OpsetImports() : opsets_;
    const int version = GetOpsetVersion(opset, domain_);
    const onnx::OpSchema* schema = onnx::OpSchemaRegistry::Schema(OpTypeToString(op_type()), version, domain_);

    // TODO(take-cheeze): Return false when schema not found
    if (!schema) {
        CLOG() << "schema of " << op_type() << "-" << version << " (" << domain_ << ") not found in: " << name_ << std::endl;
        return true;
    }

    std::ostringstream oss;
    oss << "Failed validation of " << op_type() << "-" << version << ", ";

#define output_error_message()     \
    if (message) {                 \
        *message = oss.str();      \
    } else {                       \
        CHECK(false) << oss.str(); \
    }                              \
    return false

    int min_input = schema->min_input();
    // TODO(take-cheeze): Better handler for these extensions
    if (op_type() == Node::kSequenceConstruct) {
        min_input = 0;
    }

    // TODO(take-cheeze): Input/output dtype check
    if (inputs().size() < min_input || schema->max_input() < inputs().size()) {
        oss << "Invalid input count: " << inputs().size() << " (min: " << schema->min_input() << ", max: " << schema->max_input() << ")";
        output_error_message();
    }
    if (outputs().size() < schema->min_output() || schema->max_output() < outputs().size()) {
        oss << "Invalid output count: " << outputs().size() << " (min: " << schema->min_output() << ", max: " << schema->max_output()
            << ")";
        output_error_message();
    }

    onnx::NodeProto xnode;
    ToONNX(&xnode, opsets_, false);

    for (const auto& attr_sch : schema->attributes()) {
        auto a_it = std::find_if(xnode.attribute().begin(), xnode.attribute().end(), [&attr_sch](const onnx::AttributeProto& a) {
            return attr_sch.second.name == a.name();
        });
        if (attr_sch.second.required && a_it == xnode.attribute().end()) {
            oss << "Required attribute not set: " << attr_sch.second.name;
            output_error_message();
        }
    }

#undef output_error_message

    return true;
}

std::ostream& operator<<(std::ostream& os, Node::OpType op_type) {
    os << Node::OpTypeToString(op_type);
    return os;
}

}  // namespace chainer_compiler
