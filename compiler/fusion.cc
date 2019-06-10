#include "compiler/fusion.h"

#include <limits.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <set>
#include <stack>
#include <vector>

#include <common/strutil.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/node.h>
#include <compiler/topology.h>
#include <compiler/value.h>

namespace chainer_compiler {

namespace {

void CreateFusionGroup(
        Graph* graph, const std::set<Node*>& nodes, const std::string& fusion_type, int fusion_group_id, bool can_fuse_initializers) {
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    std::vector<Value*> temps;
    ClassifyValues(std::vector<Node*>(nodes.begin(), nodes.end()), &inputs, &outputs, &temps);
    if (inputs.empty() || outputs.empty()) {
        return;
    }

    GraphBuilder gb(graph, StrCat("Fusion", fusion_group_id), outputs.front());

    auto replace_value = [&nodes](Value* value, Value* new_value) {
        if (Node* node = value->producer()) {
            if (nodes.count(node)) {
                node->ReplaceOutput(value, new_value);
            }
        }

        const std::vector<Node*> users(value->users());  // Take a copy.
        for (Node* node : users) {
            if (nodes.count(node)) {
                node->ReplaceInput(value, new_value);
            }
        }
    };

    Graph* subgraph = new Graph(StrCat("Fusion_", fusion_group_id));
    std::vector<Value*> subgraph_inputs;
    for (Value* value : inputs) {
        Value* new_value = subgraph->AddInputValue("fi_" + value->name(), value->type());
        replace_value(value, new_value);

        if (value->initializer() && value->users().empty()) {
            new_value->ResetInitializer(std::make_unique<Tensor>("fi_" + value->name(), *value->initializer()));
        } else {
            if (value->initializer()) {
                WARN_ONCE(StrCat(fusion_type, " fusion: moving initializers used more than once is not supported yet"));
            }
            subgraph_inputs.push_back(value);
        }
    }
    for (Value* value : outputs) {
        Value* new_value = subgraph->AddOutputValue("fo_" + value->name(), value->type());
        replace_value(value, new_value);
    }

    Node* fused = gb.MOp(Node::kChainerFusionGroup, subgraph_inputs, outputs);
    graph->MigrateNodes({nodes.begin(), nodes.end()}, temps, subgraph);
    fused->set_subgraph(subgraph);
    fused->set_fusion_type(fusion_type);
    fused->set_chainer_fusion_group(fusion_group_id);

#if 0
    std::cerr << "Neighbors of " << fused->ToString() << ":" << std::endl;
    for (Value* v : inputs) {
        if (v->producer()) {
            std::cerr << v->producer()->ToString() << std::endl;
        }
    }
    for (Value* v : outputs) {
        for (Node* n : v->users()) {
            std::cerr << n->ToString() << std::endl;
        }
    }
#endif
}

void RejectCyclicNodes(std::set<Node*>* cands) {
    std::stack<Node*> q;
    for (Node* node : *cands) {
        for (Value* output : node->outputs()) {
            for (Node* n : output->users()) {
                if (!cands->count(n)) q.push(n);
            }
        }
    }

    std::set<Node*> rejected;
    std::set<Node*> seen;

    while (!q.empty()) {
        Node* node = q.top();
        q.pop();
        if (!seen.emplace(node).second) continue;
        if (cands->count(node)) {
            rejected.insert(node);
        }

        // TODO(hamaji): Optimize this algorithm by pre-calculating
        // the max distance from the input for all nodes.

        for (Value* output : node->outputs()) {
            for (Node* n : output->users()) {
                q.push(n);
            }
        }
    }

    for (Node* node : rejected) cands->erase(node);
}

void RejectUnusedConstants(std::set<Node*>* cands) {
    std::set<Node*> rejected;
    for (Node* node : *cands) {
        if (node->op_type() != Node::kConstant) {
            continue;
        }
        bool is_used = false;
        for (Node* user : node->output(0)->users()) {
            if (cands->count(user)) {
                is_used = true;
                break;
            }
        }
        if (!is_used) {
            CHECK(rejected.insert(node).second);
        }
    }

    for (Node* node : rejected) cands->erase(node);
}

void FuseAllConnectedNodes(
        const char* name, Graph* graph, int min_fuse_ops, bool can_fuse_initializers, const std::function<bool(const Node&)>& is_fusable) {
    int num_fusion_groups = 0;
    const std::vector<Node*> all_nodes(graph->nodes());
    for (Node* base_node : all_nodes) {
        if (base_node->chainer_fusion_group()) continue;
        if (!is_fusable(*base_node)) continue;

        std::set<Node*> cands;
        std::stack<Node*> q;
        q.push(base_node);
        while (!q.empty()) {
            Node* node = q.top();
            CHECK_EQ(0, node->chainer_fusion_group());
            q.pop();
            if (!cands.emplace(node).second) continue;

            for (Value* value : node->inputs()) {
                Node* next_node = value->producer();
                if (!next_node) continue;
                if (!is_fusable(*next_node)) continue;
                if (base_node->IsGradNode() != next_node->IsGradNode()) continue;
                q.push(next_node);
            }
            for (Value* value : node->outputs()) {
                for (Node* next_node : value->users()) {
                    if (!is_fusable(*next_node)) continue;
                    if (base_node->IsGradNode() != next_node->IsGradNode()) continue;
                    q.push(next_node);
                }
            }
        }

        RejectCyclicNodes(&cands);
        RejectUnusedConstants(&cands);

        int num_calculation = 0;
        for (Node* node : cands) {
            if (!node->IsZeroCost()) ++num_calculation;
        }
        if (num_calculation < min_fuse_ops) continue;

        ++num_fusion_groups;
        for (Node* node : cands) {
            node->set_chainer_fusion_group(num_fusion_groups);
        }

        CreateFusionGroup(graph, cands, name, num_fusion_groups, can_fuse_initializers);
    }
}

void FuseDldtOperations(Graph* graph) {
    // The list was created by
    // $ grep 'op =' dldt/model-optimizer/extensions/front/onnx/*.py
    const std::set<Node::OpType> fusable_ops = {
            Node::kAdd,
            // Node::kAffine,
            Node::kArgMax,
            Node::kAveragePool,
            Node::kCast,
            Node::kClip,
            Node::kConstantFill,
            Node::kConv,
            Node::kConvTranspose,
            // Node::kCrop,
            // Node::kDetectionOutput,
            Node::kDropout,
            Node::kElu,
            Node::kExp,
            // Node::kExperimentalDetectronDetectionOutput,
            // Node::kExperimentalDetectronGenerateProposalsSingleImage,
            // Node::kExperimentalDetectronPriorGridGenerator,
            // Node::kExperimentalDetectronROIFeatureExtractor,
            // Node::kExperimentalDetectronTopKROIs,
            Node::kFlatten,
            Node::kGRU,
            Node::kGather,
            Node::kGlobalAveragePool,
            Node::kGlobalMaxPool,
            Node::kImageScaler,
            // Node::kInstanceNormalization,
            Node::kLRN,
            Node::kLSTM,
            Node::kLeakyRelu,
            Node::kMatMul,
            Node::kMaxPool,
            Node::kMul,
            Node::kNeg,
            Node::kPad,
            Node::kPow,
            // Node::kPriorBox,
            // Node::kQuantize,
            Node::kRNN,
            Node::kReduceMean,
            Node::kReduceSum,
            // Node::kScale,
            Node::kSigmoid,
            Node::kSlice,
            Node::kSoftmax,
            Node::kSplit,
            Node::kSqueeze,
            Node::kTanh,
            Node::kTranspose,
            Node::kUnsqueeze,
            Node::kUpsample,
    };

    auto is_fusable = [&fusable_ops](const Node& node) {
        if (!fusable_ops.count(node.op_type())) {
            return false;
        }
        for (Value* value : node.inputs()) {
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node.outputs()) {
            if (!value->type().HasKnownShape()) return false;
        }
        return true;
    };

    FuseAllConnectedNodes("dldt", graph, 1, true, is_fusable);
}

void FuseNGraphOperations(Graph* graph) {
    // TODO(hamaji): Enable all ops.
    const std::set<Node::OpType> fusable_ops = {
            Node::kAbs,
            Node::kAcos,
            Node::kAcosh,
            Node::kAdd,
            Node::kAnd,
            Node::kArgMax,
            Node::kArgMin,
            Node::kAsin,
            Node::kAsinh,
            Node::kAtan,
            Node::kAtanh,
            Node::kAveragePool,
            Node::kBatchNormalization,
            Node::kCeil,
            Node::kClip,
            Node::kConcat,
            Node::kConstant,
            Node::kConv,
            Node::kConvTranspose,
            Node::kCos,
            Node::kCosh,
            Node::kDiv,
            Node::kDropout,
            Node::kElu,
            Node::kEqual,
            Node::kExp,
            Node::kFlatten,
            Node::kFloor,
            Node::kGemm,
            Node::kGlobalAveragePool,
            // Not supported yet.
            // Node::kGlobalLpPool,
            Node::kGlobalMaxPool,
            Node::kGreater,
            // Not supported yet.
            // Node::kHardSigmoid,
            Node::kIdentity,
            // There seem to be some restrictions:
            // terminate called after throwing an instance of 'ngraph::NodeValidationFailure'
            // what():  Check '(input_shape.rank().is_dynamic() || static_cast<size_t>(input_shape.rank()) >= 3)'
            // Node::kLRN,
            Node::kLeakyRelu,
            Node::kLess,
            Node::kLog,
            Node::kLogSoftmax,
            Node::kMatMul,
            Node::kMax,
            Node::kMaxPool,
            Node::kMean,
            Node::kMin,
            Node::kMul,
            Node::kNeg,
            Node::kNot,
            // Constant input only.
            // Node::kOneHot,
            Node::kOr,
            // Not supported yet.
            // Node::kPRelu,
            Node::kPow,
            Node::kReciprocal,
            Node::kReduceL1,
            Node::kReduceL2,
            Node::kReduceLogSum,
            Node::kReduceLogSumExp,
            Node::kReduceMax,
            Node::kReduceMean,
            Node::kReduceMin,
            // Not supported yet.
            // Node::kReduceProd,
            Node::kReduceSum,
            Node::kReduceSumSquare,
            Node::kRelu,
            Node::kSelu,
            Node::kShape,
            Node::kSigmoid,
            Node::kSign,
            Node::kSin,
            Node::kSinh,
            Node::kSize,
            Node::kSlice,
            Node::kSoftmax,
            Node::kSoftplus,
            Node::kSoftsign,
            Node::kSplit,
            Node::kSqrt,
            Node::kSqueeze,
            Node::kSub,
            Node::kSum,
            Node::kTan,
            Node::kTanh,
            // Not supported yet.
            // Node::kTopK,
            Node::kTranspose,
            Node::kUnsqueeze,
            Node::kXor,
            Node::kWhere,
            Node::kPad,
            Node::kReshape,
    };

    auto is_fusable = [&fusable_ops](const Node& node) {
        if (!fusable_ops.count(node.op_type())) {
            return false;
        }
        for (Value* value : node.inputs()) {
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node.outputs()) {
            if (!value->type().HasKnownShape()) return false;
        }

        if (node.op_type() == Node::kReshape) {
            CHECK_EQ(2, node.inputs().size());
            if (!node.input(1)->producer() || node.input(1)->producer()->op_type() != Node::kConstant) {
                return false;
            }
        } else if (node.op_type() == Node::kMaxPool) {
            if (node.chainer_cover_all()) {
                return false;
            }
        } else if (
                node.op_type() == Node::kAdd || node.op_type() == Node::kSub || node.op_type() == Node::kMul ||
                node.op_type() == Node::kDiv || node.op_type() == Node::kPow) {
            // No type coercion in nGraph.
            if (node.input(0)->type().dtype() != node.input(1)->type().dtype()) {
                return false;
            }
        } else if (node.op_type() == Node::kPad) {
            // Apparently, nGraph does not support negative pads.
            for (int p : node.pads()) {
                if (p < 0) {
                    return false;
                }
            }
        } else if (node.op_type() == Node::kTranspose) {
            // Incomplete transpose is our own extension to ONNX.
            if (!node.perm().empty() && node.input(0)->type().ndim() != node.perm().size()) {
                return false;
            }
        } else if (node.op_type() == Node::kBatchNormalization) {
            // nGraph does not support BatchNorm in training mode.
            if (node.outputs().size() != 1) {
                return false;
            }
        } else if (node.op_type() == Node::kSlice) {
            // nGraph does not support new slice.
            if (node.inputs().size() > 1) {
                return false;
            }
        } else if (node.op_type() == Node::kSoftmax || node.op_type() == Node::kLogSoftmax) {
            // nGraph does not know Chainer's Softmax.
            if (!node.chainer_is_onnx_semantics()) {
                return false;
            }
        }

        return true;
    };

    FuseAllConnectedNodes("ngraph", graph, 1, false, is_fusable);
}

void FuseTVMOperations(Graph* graph) {
    auto is_fusable = [](Node* node) {
        for (Value* value : node->inputs()) {
            if (value->type().dtype() == Dtype::kInt64) return false;
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node->outputs()) {
            if (value->type().dtype() == Dtype::kInt64) return false;
            if (!value->type().HasKnownShape()) return false;
        }
        return true;
    };

    int num_fusion_groups = 0;
    std::set<Node*> handled;
    for (Node* base_node : graph->GetTopologicallySortedNodes()) {
        if (base_node->op_type() != Node::kRelu && base_node->op_type() != Node::kTanh && base_node->op_type() != Node::kConv &&
            base_node->op_type() != Node::kConvTranspose) {
            continue;
        }
        if (!handled.emplace(base_node).second) {
            continue;
        }
        if (!is_fusable(base_node)) {
            continue;
        }

        std::set<Node*> fused_nodes = {base_node};

        Node* node = base_node;
        while (true) {
            CHECK_EQ(1, node->outputs().size());
            Value* output = node->output(0);
            if (output->users().size() != 1) {
                break;
            }

            Node* user = output->user(0);
            if ((user->op_type() != Node::kRelu && user->op_type() != Node::kReduceSum && user->op_type() != Node::kAdd)) {
                break;
            }
            if (!handled.emplace(user).second) {
                break;
            }
            if (!is_fusable(user)) {
                break;
            }
            CHECK(fused_nodes.emplace(user).second);
            node = user;
        }

        int num_calculation = 0;
        for (Node* node : fused_nodes) {
            if (node->op_type() != Node::kIdentity && node->op_type() != Node::kConstant) ++num_calculation;
        }
        if (num_calculation <= 1 && base_node->op_type() != Node::kConv && base_node->op_type() != Node::kConvTranspose) {
            continue;
        }

        ++num_fusion_groups;
        for (Node* node : fused_nodes) {
            node->set_chainer_fusion_group(num_fusion_groups);
        }
        CreateFusionGroup(graph, fused_nodes, "tvm", false, num_fusion_groups);
    }
}

void FuseElementwiseOperations(Graph* graph) {
    // TODO(hamaji): Do not try fusing integer ops.
    const std::set<Node::OpType> fusable_ops = {
            Node::kIdentity,
            Node::kAdd,
            Node::kSub,
            Node::kMul,
            // Node::kDiv,
            Node::kTanh,
            Node::kSigmoid,
            Node::kExp,
    };

    auto is_fusable = [&fusable_ops](const Node& node) {
        if (node.op_type() == Node::kConstant) {
            Tensor* t = node.tensor_value().get();
            return t->dtype().IsFloat() && t->NumElements() == 1;
        }

        if (!fusable_ops.count(node.op_type())) return false;
        for (Value* value : node.inputs()) {
            Dtype dtype = value->type().dtype();
            // TODO(hamaji): Fix the dtype inference and do not fuse
            // unknown dtypes.
            if (!dtype.IsFloat() && dtype != Dtype::kUnknown) return false;
        }
        return true;
    };

    FuseAllConnectedNodes("nvrtc", graph, 2, false, is_fusable);
}

}  // namespace

void FuseOperations(Graph* graph) {
    // Fuse ops in subgraphs first to avoid infinite loop.
    for (const Node* node : graph->nodes()) {
        for (Graph* subgraph : node->GetSubGraphs()) {
            FuseOperations(subgraph);
        }
    }

    if (g_use_dldt) {
        FuseDldtOperations(graph);
    }
    if (g_use_ngraph) {
        FuseNGraphOperations(graph);
    }
    if (g_use_tvm) {
        FuseTVMOperations(graph);
    }
    if (g_fuse_operations) {
        FuseElementwiseOperations(graph);
    }
}

}  // namespace chainer_compiler
