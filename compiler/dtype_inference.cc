#include "compiler/dtype_inference.h"

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>
#include <compiler/value.h>

namespace chainer_compiler {

Dtype CoerceDtype(Dtype dtype0, Dtype dtype1) {
    if (dtype0 == dtype1) return dtype0;
    if (dtype0 == Dtype::kUnknown || dtype1 == Dtype::kUnknown) return Dtype::kUnknown;
    if (dtype0.IsFloat() && !dtype1.IsFloat()) return dtype0;
    if (!dtype0.IsFloat() && dtype1.IsFloat()) return dtype1;
    if (dtype0.SizeOf() > dtype1.SizeOf()) return dtype0;
    if (dtype0.SizeOf() < dtype1.SizeOf()) return dtype1;
    if (dtype1 == Dtype::kBool) return dtype0;
    if (dtype0 == Dtype::kBool) return dtype1;
    if (dtype0 == Dtype::kUInt8 || dtype1 == Dtype::kUInt8) return Dtype::kInt16;
    CHECK(false) << "Unknown type coerce: " << dtype0.ToString() << " vs " << dtype1.ToString();
}

void InferDtype(Node* node) {
    Dtype default_float = Dtype(Dtype::kFloat32);

    auto coerce = [node]() {
        const std::vector<Value*>& ins = node->inputs();
        Dtype dtype = ins[0]->type().dtype();
        for (size_t i = 1; i < ins.size(); ++i) {
            dtype = CoerceDtype(dtype, ins[i]->type().dtype());
        }
        return dtype;
    };

    auto set = [node](int i, Dtype dtype) {
        CHECK_LT(i, node->outputs().size());
        Dtype odtype = node->output(i)->type().dtype();
        if (odtype == Dtype::kUnknown) {
            node->output(i)->mutable_type()->set_dtype(dtype);
        } else {
            if (dtype != Dtype::kUnknown) CHECK_EQ(dtype, odtype) << "dtype mismatch for output #" << i << " of " << node->ToString();
        }
    };

    auto oset = [node, set](int i, Dtype dtype) {
        if (i < node->outputs().size()) set(i, dtype);
    };

    Dtype in0 = Dtype::kUnknown;
    if (node->inputs().size() >= 1) in0 = node->input(0)->type().dtype();
    Dtype in1 = Dtype::kUnknown;
    if (node->inputs().size() >= 2) in1 = node->input(1)->type().dtype();
    Dtype in2 = Dtype::kUnknown;
    if (node->inputs().size() >= 3) in2 = node->input(2)->type().dtype();

    switch (node->op_type()) {
        case Node::kReciprocal:
        case Node::kExp:
        case Node::kSin:
        case Node::kSinh:
        case Node::kCos:
        case Node::kCosh:
        case Node::kTan:
        case Node::kTanh:
        case Node::kAsin:
        case Node::kAsinh:
        case Node::kAcos:
        case Node::kAcosh:
        case Node::kAtan:
        case Node::kAtanh:
        case Node::kLog:
        case Node::kSqrt:
        case Node::kSigmoid:
        case Node::kSelu:
        case Node::kLeakyRelu:
        case Node::kElu:
        case Node::kSoftsign:
        case Node::kSoftplus:
        case Node::kReduceMean:
        case Node::kHardmax:
        case Node::kDropout:
        case Node::kLRN:
        case Node::kLpNormalization:
        case Node::kSoftmax:
        case Node::kLogSoftmax:
        case Node::kAveragePool:
        case Node::kGlobalAveragePool: {
            set(0, in0.IsFloat() ? in0 : default_float);
            break;
        }

        case Node::kConstantFill: {
            set(0, node->dtype());
            break;
        }

        case Node::kConstantLike: {
            set(0, node->dtype());
            break;
        }

        case Node::kCast: {
            set(0, node->to());
            break;
        }

        case Node::kEyeLike: {
            if (node->dtype()) {
                set(0, node->dtype());
            } else {
                set(0, in0 == Dtype::kUnknown ? default_float : in0);
            }
            break;
        }

        case Node::kAdd:
        case Node::kSub:
        case Node::kMul:
        case Node::kDiv:
        case Node::kPow:
        case Node::kSum:
        case Node::kMean:
        case Node::kMax:
        case Node::kMin:
        case Node::kConcat:
        case Node::kMatMul:
        case Node::kGemm:
        case Node::kChainerLinear:
        case Node::kChainerLinearGradWeight: {
            set(0, coerce());
            break;
        }

        case Node::kNot:
        case Node::kEqual:
        case Node::kGreater:
        case Node::kLess:
        case Node::kAnd:
        case Node::kOr:
        case Node::kXor: {
            set(0, Dtype::kBool);
            break;
        }

        case Node::kArgMax:
        case Node::kArgMin:
        case Node::kSize:
        case Node::kShape: {
            set(0, Dtype::kInt64);
            break;
        }

        case Node::kConstant: {
            set(0, node->tensor_value()->dtype());
            break;
        }

        case Node::kConstantOfShape: {
            set(0, node->tensor_value() ? node->tensor_value()->dtype() : default_float);
            break;
        }

        case Node::kRNN:
        case Node::kGRU:
        case Node::kLSTM: {
            Dtype dtype = CoerceDtype(in0, in1);
            if (node->inputs().size() >= 3) dtype = CoerceDtype(dtype, node->input(2)->type().dtype());
            oset(0, dtype);
            oset(1, dtype);
            oset(2, dtype);
            break;
        }

        case Node::kConv:
        case Node::kConvTranspose:
        case Node::kChainerConvGradWeight: {
            Dtype dtype = CoerceDtype(in0, in1);
            if (node->inputs().size() >= 3) dtype = CoerceDtype(dtype, node->input(2)->type().dtype());
            oset(0, dtype);
            break;
        }

        case Node::kMaxRoiPool:
        case Node::kChainerROIMaxPool2D:
        case Node::kChainerROIAveragePool2D:
        case Node::kChainerROIMaxAlign2D:
        case Node::kChainerROIAverageAlign2D:
        case Node::kChainerResizeImages:
        case Node::kChainerMaxPoolGrad:
        case Node::kChainerAveragePoolGrad:
        case Node::kChainerReluGrad:
        case Node::kChainerLRNGrad: {
            set(0, coerce());
            break;
        }

        case Node::kChainerBatchNormalizationGrad: {
            Dtype dtype = coerce();
            set(0, dtype);
            set(1, dtype);
            set(2, dtype);
            break;
        }

        case Node::kChainerConvTransposeWithDynamicOutputShape: {
            CHECK(in2 == Dtype::kInt64 || in2 == Dtype::kUnknown) << in1.ToString() << " in " << node->ToString();
            set(0, CoerceDtype(in0, in1));
            break;
        }

        default: {
            // TODO(hamaji): Consider revive dtype inference.
            break;
        }
    }
}

void InferAllDtype(Graph* graph) {
    for (Node* node : graph->GetTopologicallySortedNodes()) {
        InferDtype(node);
    }
}

}  // namespace chainer_compiler
