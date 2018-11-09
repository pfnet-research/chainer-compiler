#include "nvrtc_builder.h"

#include <ctype.h>

#include <algorithm>
#include <iterator>
#include <queue>
#include <map>
#include <set>
#include <sstream>

#include <common/log.h>
#include <compiler/code_emitter.h>
#include <compiler/node.h>
#include <compiler/value.h>

namespace oniku {

namespace {

std::string CleanseIdent(const std::string& s, const char* prefix = "v_") {
    std::string o = prefix;
    for (char c : s) {
        if (std::isalnum(c)) {
            o += c;
        } else {
            o += '_';
        }
    }
    return o;
}

void EmitNode(const Node* node, CodeEmitter* ce) {
    std::vector<std::string> ins;
    std::vector<std::string> outs;
    for (Value* value : node->inputs()) ins.push_back(CleanseIdent(value->name()));
    for (Value* value : node->outputs()) outs.push_back(CleanseIdent(value->name()));

    auto out1 = [&outs, node, ce](const std::string& rhs) {
        CHECK_EQ(1UL, outs.size());
        *ce << "T " << outs[0] << " = " << rhs << ";  // " << node->op_type() << "\n";
    };

    auto binary = [&ins, out1](char op) {
        CHECK_EQ(2UL, ins.size());
        out1(ins[0] + ' ' + op + ' ' + ins[1]);
    };

    switch (node->op_type()) {
    case Node::kIdentity:
        out1(ins[0]);
        break;

    case Node::kTanh:
        out1("tanh(" + ins[0] + ")");
        break;

    case Node::kSigmoid:
        out1("sigmoid(" + ins[0] + ")");
        break;

    case Node::kAdd:
        binary('+');
        break;

    case Node::kSub:
        binary('-');
        break;

    case Node::kMul:
        binary('*');
        break;

    case Node::kDiv:
        binary('/');
        break;

    default:
        CHECK(false) << "Cannot build NVRTC program for: " << node->ToString();
    }
}

void FindInOuts(const std::vector<Node*>& nodes, std::vector<Value*>* inputs, std::vector<Value*>* outputs) {
    std::set<Value*> input_set;
    std::map<Value*, int> output_users;
    for (Node* node : nodes) {
        for (Value* value : node->inputs()) {
            input_set.insert(value);
        }
        for (Value* value : node->outputs()) {
            CHECK(output_users.emplace(value, value->users().size()).second);
        }
    }

    for (Node* node : nodes) {
        for (Value* value : node->inputs()) {
            auto found = output_users.find(value);
            if (found != output_users.end()) {
                --found->second;
            }
        }
        for (const auto& p : output_users) {
            input_set.erase(p.first);
        }
    }

    inputs->assign(input_set.begin(), input_set.end());
    for (const auto& p : output_users) {
        CHECK_LE(0, p.second);
        if (p.second > 0) outputs->push_back(p.first);
    }
}

}  // namespace

void BuildNvrtcProgram(const std::vector<Node*>& nodes,
                       int id,
                       std::string* prog,
                       std::vector<Value*>* inputs,
                       std::vector<Value*>* outputs) {
    std::set<Node::OpType> seen_ops;
    for (Node* node : nodes) {
        seen_ops.insert(node->op_type());
    }
    FindInOuts(nodes, inputs, outputs);

    std::ostringstream oss;
    CodeEmitter ce(oss);
    ce << "typedef float T;\n";
    if (seen_ops.count(Node::kSigmoid)) {
        ce << "__device__ T sigmoid(T x) {\n";
        ce << "const T half = 0.5;\n";
        ce << "return tanh(x * half) * half + half;\n";
        ce << "}\n";
    }

    ce << "extern \"C\" __global__\n";
    ce << "void fusion" << id << "(size_t n";
    for (Value* value : *inputs) {
        ce << ", T* " << CleanseIdent(value->name(), "i_");
    }
    for (Value* value : *outputs) {
        ce << ", T* " << CleanseIdent(value->name(), "o_");
    }
    ce << ") {\n";
    ce << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ce << "if (tid >= n) return;\n";
    for (Value* value : *inputs) {
        ce << "T " << CleanseIdent(value->name()) << " = " << CleanseIdent(value->name(), "i_") << "[tid];  // input\n";
    }

    std::map<Node*, int> input_counts;
    for (Node* node : nodes) {
        CHECK(input_counts.emplace(node, node->GetNumActualInputs()).second);
    }

    std::queue<Value*> q;
    for (Value* value : *inputs) {
        q.push(value);
    }

    while (!q.empty()) {
        Value* value = q.front();
        q.pop();

        for (Node* node : value->users()) {
            auto found = input_counts.find(node);
            if (found == input_counts.end()) continue;
            if (--found->second != 0) continue;
            EmitNode(node, &ce);
            for (Value* value : node->outputs()) q.push(value);
        }
    }

    for (Value* value : *outputs) {
        ce << CleanseIdent(value->name(), "o_") << "[tid] = " << CleanseIdent(value->name()) << ";  // output\n";
    }

    ce << "}\n";

    *prog = oss.str();
}

}
