#include "compiler/graph.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <queue>
#include <set>

#include <compiler/onnx.h>
#include <onnx/shape_inference/implementation.h>

#include <common/log.h>
#include <common/strutil.h>
#include <compiler/node.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/topology.h>
#include <compiler/util.h>
#include <compiler/value.h>

namespace chainer_compiler {

Graph::Graph(const OpsetList& opsets, const onnx::GraphProto& xgraph) : opset_import_(opsets) {
    Construct(xgraph);
}

void Graph::Construct(const onnx::GraphProto& xgraph) {
    name_ = xgraph.name();
    doc_string_ = xgraph.doc_string();
    std::map<std::string, Value*> values_by_name;
    for (const onnx::ValueInfoProto& input : xgraph.input()) {
        Value* value = new Value(input, Value::Kind::kInput);
        all_values_.emplace_back(value);
        input_values_.push_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& output : xgraph.output()) {
        std::unique_ptr<Value> value(new Value(output, Value::Kind::kOutput));
        auto p = values_by_name.emplace(value->name(), value.get());
        if (p.second) {
            output_values_.push_back(value.get());
            all_values_.emplace_back(std::move(value));
        } else {
            // We allow graph output to be null.
            // TODO(hamaji): Revisit this design. Probably, it would
            // be better to mark outputs are unnecessary instead of
            // using null values.
            CHECK(value->name().empty()) << "Duplicated value name: " << value->name();
            output_values_.push_back(p.first->second);
        }
    }
    for (const onnx::ValueInfoProto& temp : xgraph.value_info()) {
        std::unique_ptr<Value> value(new Value(temp, Value::Kind::kTemp));
        auto p = values_by_name.emplace(value->name(), value.get());
        if (!p.second) {
            // Allow having both output and value_info for old torch exporter.
            CHECK_EQ(value->type().DebugString(), p.first->second->type().DebugString())
                    << "Duplicated value name with different type: " << value->ToString();
            continue;
        }
        temp_values_.push_back(value.get());
        all_values_.emplace_back(std::move(value));
    }

    for (const onnx::TensorProto& xtensor : xgraph.initializer()) {
        std::unique_ptr<Tensor> tensor(new Tensor(xtensor));
        auto found = values_by_name.find(tensor->name());
        if (found != values_by_name.end()) {
            CHECK(found->second->IsInput()) << "Only input can have an initializer but " << found->second->DebugString();
            found->second->ResetInitializer(std::move(tensor));
        } else {
            Type type(tensor->dtype(), tensor->dims());
            Value* value = new Value(tensor->name(), type, Value::kInput);
            value->ResetInitializer(std::move(tensor));
            all_values_.emplace_back(value);
            input_values_.push_back(value);
            CHECK(values_by_name.emplace(value->name(), value).second);
        }
    }

    auto get_value = [&](const std::string& name) {
        auto p = values_by_name.emplace(name, nullptr);
        if (!p.second) return p.first->second;
        return p.first->second = AddValue(name);
    };

    for (const onnx::NodeProto& xnode : xgraph.node()) {
        std::vector<Value*> inputs;
        for (const std::string& name : xnode.input()) {
            inputs.push_back(get_value(name));
        }
        std::vector<Value*> outputs;
        for (const std::string& name : xnode.output()) {
            outputs.push_back(get_value(name));
        }

        Node* node = new Node(opset_import_, xnode, inputs, outputs);
        // TODO(take-cheeze): ONNX should support undefined value case
        switch (node->op_type()) {
            case Node::kSequenceConstruct:
            case Node::kSequenceErase:
            case Node::kSequenceInsert:
            case Node::kSplitToSequence: {
                Value& out = *node->output(0);
                const Type& t = out.type();
                if (t.kind() == Type::Kind::kTensor && t.dtype() == Dtype::kUnknown) {
                    out.set_type(new Type(Type::Kind::kSequence));
                }
            } break;

            default:
                break;
        }
        AddNodeImpl(std::unique_ptr<Node>(node), inputs, outputs);
    }
}

Graph::Graph(const OpsetList& opsets, const std::string name) : name_(name), opset_import_(opsets) {
}

Graph::~Graph() {
}

void Graph::ToONNX(onnx::GraphProto* xgraph, bool serialize_initializers) const {
    DUMP_STRING(xgraph, name);
    DUMP_STRING(xgraph, doc_string);

    // `temp_values_` may contain kInput and kOutput due to `ChangeKind`.
    // TODO(hamaji): Remove `ChangeKind` by refactoring how gradients
    // of subgraphs are created.
    std::set<Value*> input_values{input_values_.begin(), input_values_.end()};
    std::set<Value*> output_values{output_values_.begin(), output_values_.end()};
    for (const auto& value : all_values_) {
        onnx::ValueInfoProto* xvalue = nullptr;
        if (input_values.count(value.get())) {
            xvalue = xgraph->add_input();
        } else if (output_values.count(value.get())) {
            xvalue = xgraph->add_output();
        } else if (!value->IsNull()) {
            xvalue = xgraph->add_value_info();
        }
        if (!xvalue) continue;

        value->ToONNX(xvalue);
        if (serialize_initializers) {
            if (const Tensor* initializer = value->initializer()) {
                onnx::TensorProto* xtensor = xgraph->add_initializer();
                initializer->ToONNX(xtensor);
            }
        }
    }

    for (const Node* node : nodes_) {
        onnx::NodeProto* xnode = xgraph->add_node();
        node->ToONNX(xnode, opset_import_);
    }
}

std::string Graph::DebugString() const {
    onnx::GraphProto xgraph;
    ToONNX(&xgraph);
    StripONNXGraph(&xgraph);
    return xgraph.DebugString();
}

std::vector<Node*> Graph::GetLiveNodes() const {
    std::vector<Node*> nodes;
    for (Node* node : nodes_) {
        if (!node->detached()) nodes.push_back(node);
    }
    return nodes;
}

std::set<Value*> Graph::GetNecessaryValues(const std::vector<Value*>& output_values) const {
    std::queue<Value*> q;
    for (Value* value : output_values) q.push(value);

    std::set<Value*> seen_values;
    while (!q.empty()) {
        Value* value = q.front();
        q.pop();
        if (Node* node = value->producer()) {
            for (Value* input : node->inputs()) {
                if (!seen_values.emplace(input).second) continue;
                q.push(input);
            }
        }
    }
    return seen_values;
}

std::set<Value*> Graph::GetNecessaryValues() const {
    return GetNecessaryValues(output_values_);
}

Value* Graph::AddValue(const std::string& name, const Type& type, Value::Kind kind) {
    Value* value = new Value(MakeUnique(name), type, kind);
    all_values_.emplace_back(value);
    if (value->IsInput()) input_values_.push_back(value);
    if (value->IsOutput()) output_values_.push_back(value);
    if (value->IsTemp()) temp_values_.push_back(value);
    return value;
}

Value* Graph::AddValue(const std::string& name, Value::Kind kind) {
    return AddValue(name, Type(), kind);
}

Value* Graph::AddInputValue(const std::string& name, const Type& type) {
    return AddValue(name, type, Value::Kind::kInput);
}

Value* Graph::AddOutputValue(const std::string& name, const Type& type, int index) {
    Value* value = AddValue(name, type, Value::Kind::kOutput);
    if (index >= 0) {
        output_values_.pop_back();
        CHECK_LE(index, output_values_.size());
        output_values_.insert(output_values_.begin() + index, value);
    }
    return value;
}

Value* Graph::AddNullValue() {
    return AddValue("", Value::Kind::kNull);
}

void Graph::ResetKind(Value* value) {
    CHECK(!value->IsTemp()) << value->ToString();
    if (value->IsInput()) {
        auto found = std::find(input_values_.begin(), input_values_.end(), value);
        CHECK(found != input_values_.end()) << value->ToString();
        input_values_.erase(found);
    }
    if (value->IsOutput()) {
        auto found = std::find(output_values_.begin(), output_values_.end(), value);
        CHECK(found != output_values_.end()) << value->ToString();
        output_values_.erase(found);
    }
    value->kind_ = Value::Kind::kTemp;
    temp_values_.push_back(value);
}

Node* Graph::AddNode(
        Node::OpType op_type,
        const std::vector<Value*>& inputs,
        const std::vector<Value*>& outputs,
        const std::string& base,
        const std::string& domain,
        const OpsetList& opsets) {
    Node* node = new Node(GenSym(base.empty() ? Node::OpTypeToString(op_type) : base), op_type, inputs, outputs, domain, opsets);
    AddNodeImpl(std::unique_ptr<Node>(node), inputs, outputs);
    return node;
}

Node* Graph::AddNode(
        const onnx::NodeProto& base, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, const std::string& name) {
    Node* node = new Node(opset_import_, base, inputs, outputs, name);
    AddNodeImpl(std::unique_ptr<Node>(node), inputs, outputs);
    return node;
}

void Graph::DetachNode(Node* node) {
    node->Detach();
}

std::vector<Node*> Graph::GetTopologicallySortedNodes() const {
    return SortTopologically(GetLiveNodes(), input_values(), true);
}

void Graph::SortNodesTopologically() {
    std::set<Node*> node_set{nodes_.begin(), nodes_.end()};
    std::vector<Node*> next_nodes = GetTopologicallySortedNodes();
    for (Node* node : next_nodes) {
        CHECK(node_set.erase(node));
    }
    for (Node* node : node_set) {
        next_nodes.push_back(node);
    }
    nodes_.swap(next_nodes);
}

std::vector<std::pair<Value*, int>> Graph::GetTopologicallySortedValuesWithDistance() const {
    return SortValuesTopologicallyWithDistance(GetLiveNodes(), input_values(), true);
}

std::map<Node*, int> Graph::GetNecessaryNodesAndInputCounts(const std::vector<Value*>& output_values) const {
    std::queue<Node*> q;
    for (const Value* value : output_values) {
        q.push(value->producer());
    }

    // Nodes without any outputs are always necessary (e.g., ChainerPrint).
    for (Node* node : nodes_) {
        if (node->outputs().empty()) {
            q.push(node);
        }
    }

    // All node in this graph for sanity check.
    std::set<Node*> node_set(nodes_.begin(), nodes_.end());

    std::map<Node*, int> input_counts;
    while (!q.empty()) {
        Node* node = q.front();
        q.pop();
        if (!node) continue;
        if (!input_counts.emplace(node, node->GetNumActualInputs()).second) continue;
        if (!node_set.count(node)) {
            std::cerr << "External reference from " << this->name() << ". External node:\n" << node->DebugString();
            DumpONNXOnFailure();
            CHECK(false);
        }

        for (const Value* input : node->inputs()) {
            q.push(input->producer());
            for (Node* node : input->users()) {
                if (node->outputs().empty()) q.push(node);
            }
        }
    }
    return input_counts;
}

std::vector<const Node*> Graph::GetComputationSequence() const {
    std::vector<const Node*> nodes;
    for (const Node* node : nodes_) {
        if (node->chainer_order() >= 0) nodes.push_back(node);
    }
    std::sort(nodes.begin(), nodes.end(), [](const Node* a, const Node* b) { return a->chainer_order() < b->chainer_order(); });
    return nodes;
}

std::string Graph::GenSym(const std::string& base) {
    std::ostringstream oss;
    if (!base.empty()) oss << base << "_";
    oss << "gensym";
    return MakeUnique(oss.str());
}

std::string Graph::MakeUnique(const std::string& name) {
    if (name.empty()) return name;
    int id = ids_[name]++;
    if (id == 0) return name;
    return StrCat(name, '_', id);
}

void Graph::AddNodeImpl(std::unique_ptr<Node> node, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    for (Value* input : inputs) input->AddUser(node.get());
    for (Value* output : outputs) output->SetProducer(node.get());
    nodes_.push_back(node.get());
    nodes_buf_.emplace_back(std::move(node));
}

void Graph::MigrateNodes(const std::vector<Node*>& nodes, const std::vector<Value*>& temps, Graph* to) {
    for (Node* node : nodes) {
        auto found = std::find(nodes_.begin(), nodes_.end(), node);
        CHECK(found != nodes_.end()) << node->DebugString();
        nodes_.erase(found);
        to->nodes_.push_back(node);
    }
    for (Value* value : temps) {
        auto found = std::find(temp_values_.begin(), temp_values_.end(), value);
        CHECK(found != temp_values_.end()) << value->DebugString();
        temp_values_.erase(found);
        to->temp_values_.push_back(value);
    }
    to->SortNodesTopologically();
}

void Graph::InferShapes() {
    onnx::GraphProto xgraph;
    ToONNX(&xgraph);
    output_values_.clear();
    input_values_.clear();
    temp_values_.clear();
    all_values_.clear();
    nodes_.clear();
    nodes_buf_.clear();
    std::unordered_map<std::string, int> opset_imports;
    if (opset_import_.empty()) {
        opset_imports = DefaultOpsetImports();
    } else {
        for (const auto& i : opset_import_) {
            opset_imports.insert(std::make_pair(i.domain(), i.version()));
        }
    }
    onnx::shape_inference::InferShapes(&xgraph, opset_imports);
    Construct(xgraph);
}

void Graph::ResetGradients(bool reset_grad_names) {
    for (const auto& v : all_values()) {
        if (Value* gv = v->grad()) {
            if (reset_grad_names && v->IsTemp()) {
                gv->ResetName("grad@" + v->name());
            }
            gv->set_type(new Type(v->type()));
            v->set_grad(nullptr);
        }
    }
}

void Graph::DeleteDetached() {
    nodes_ = GetLiveNodes();
    std::sort(nodes_.begin(), nodes_.end(), [](Node* l, Node* r) { return l->chainer_order() < r->chainer_order(); });
}

void Graph::CheckSanity(const std::string& msg) const {
    // Check if names of values are distinct.
    std::set<std::string> value_names;
    std::set<Value*> value_set;
    for (const auto& value : all_values_) {
        if (value->name().empty()) continue;
        if (!value_names.emplace(value->name()).second) {
            std::cerr << "ERROR: Duplicated name: " << value->name() << std::endl;
            DumpONNXOnFailure();
            CHECK(false) << msg;
        }
        CHECK(value_set.emplace(value.get()).second);
    }

    std::set<Node*> node_set;
    for (const auto& node : nodes_buf_) {
        CHECK(node_set.emplace(node.get()).second);
    }

    // Check if a value is output at most once.
    {
        bool ok = true;
        std::set<Value*> output_set;
        for (Value* value : input_values_) {
            if (!output_set.insert(value).second) {
                std::cerr << "ERROR: A value appears as input of the graph more than once: " << value->name() << std::endl;
                ok = false;
            }
        }
        for (Node* node : nodes_) {
            for (Value* value : node->outputs()) {
                if (!output_set.insert(value).second) {
                    std::cerr << "ERROR: A value is output more than once: `" << value->name() << "` creator: " << node->ToString()
                              << std::endl;
                    ok = false;
                }
            }
        }
        if (!ok) {
            DumpONNXOnFailure();
            CHECK(false) << "Sanity check (SSA) failed";
        }
    }

    // TODO(hamaji): No cycle and no links to nodes outside the graph.
}

void Graph::DumpSubGraphs(int depth) const {
    for (int i = 0; i < depth; i++) std::cerr << ' ';
    std::cerr << name() << ' ' << input_values().size() << " inputs " << output_values().size() << " outputs" << std::endl;
    for (const Node* node : nodes_) {
        for (Graph* sub_graph : node->GetSubGraphs()) {
            sub_graph->DumpSubGraphs(depth + 1);
        }
    }
}

void Graph::DumpONNXOnFailure(const std::string& filename) const {
    onnx::ModelProto xmodel;
    xmodel.set_ir_version(3);
    xmodel.set_producer_name("chainer compiler failed :(");
    ToONNX(xmodel.mutable_graph());
    const std::string fn = filename.empty() ? "/tmp/chainer_compiler_failure.onnx" : filename;
    std::ofstream ofs(fn);
    xmodel.SerializeToOstream(&ofs);
    std::cerr << "Failed graph is stored in " << fn << std::endl;
}

int Graph::MinVersion(const std::string& domain) const {
    int min = 1000;
    for (const Node* n : nodes_) {
        if (n->detached() || n->domain() != domain) {
            continue;
        }
        if (n->OpVersion() < min) {
            min = n->OpVersion();
        }
    }
    return min;
}

int Graph::MaxVersion(const std::string& domain) const {
    int max = -1;
    for (const Node* n : nodes_) {
        if (n->detached() || n->domain() != domain) {
            continue;
        }
        if (n->OpVersion() > max) {
            max = n->OpVersion();
        }
    }
    return max;
}

}  // namespace chainer_compiler
