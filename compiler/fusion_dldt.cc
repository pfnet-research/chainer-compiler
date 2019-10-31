#include <set>

#include <chainerx/array.h>
#include <chainerx/routines/manipulation.h>

#include <compiler/flags.h>
#include <compiler/fusion.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/value.h>
#include <runtime/chainerx_util.h>

namespace chainer_compiler {

void FuseDldtOperations(Graph* graph) {
    // The list was created by
    // $ grep 'op =' dldt/model-optimizer/extensions/front/onnx/*.py
    // and `onnx_op_extractors` in mo/front/onnx/extractor.py.
    std::set<Node::OpType> fusable_ops = {
            Node::kAdd,
            // Node::kAffine,
            Node::kArgMax,
            Node::kAveragePool,
            Node::kBatchNormalization,
            Node::kCast,
            Node::kClip,
            Node::kConcat,
            Node::kConstant,
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
            Node::kGemm,
            Node::kGlobalAveragePool,
            Node::kGlobalMaxPool,
            Node::kIdentity,
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
            Node::kRelu,
            Node::kReshape,
            // Node::kScale,
            Node::kSigmoid,
            // TODO(hamaji): We should be able to support this through
            // simplifier, but the naming scheme will be confused.
            // Node::kSlice,
            Node::kSoftmax,
            Node::kSplit,
            // TODO(hamaji): Implement simplifier.
            // Node::kSqueeze,
            Node::kSum,
            Node::kTanh,
            Node::kTranspose,
            Node::kUnsqueeze,
    };

    if (g_dldt_device == "GPU") {
        CHECK(fusable_ops.emplace(Node::kResize).second);
        CHECK(fusable_ops.emplace(Node::kUpsample).second);
    }

    auto is_fusable = [&fusable_ops](const Node& node) {
        if (!fusable_ops.count(node.op_type())) {
            return false;
        }

        if (node.op_type() == Node::kPad &&
            (!node.input(1)->GetConstTensor() || (node.inputs().size() >= 3 && !node.input(2)->GetConstTensor()))) {
            return false;
        }

        for (Value* value : node.inputs()) {
            if (!value->type().HasKnownShape()) return false;
        }
        for (Value* value : node.outputs()) {
            if (!value->type().HasKnownShape()) return false;
        }

        if (node.op_type() == Node::kResize || node.op_type() == Node::kUpsample) {
            if (node.inputs().size() != 2) {
                return false;
            }
            if (node.mode() != "nearest") {
                return false;
            }
            const Tensor* scales_tensor = node.input(1)->GetConstTensor();
            if (!scales_tensor) {
                return false;
            }
            const chainerx::Array& a = scales_tensor->chx();
            if (a.shape().size() != 1) {
                return false;
            }
            std::vector<double> scales;
            for (int64_t i = 0; i < a.GetTotalSize(); ++i) {
                scales.emplace_back(chainerx::AsScalar(a.At({i})));
            }
            if (scales.size() != 4 || scales[0] != 1 || scales[1] != 1 || scales[2] != scales[3]) {
                return false;
            }
        }

        return true;
    };

    FuseAllConnectedNodes("dldt", graph, 1, true, is_fusable);
}

}  // namespace chainer_compiler
