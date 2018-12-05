#include "dtype_inference.h"

#include <common/log.h>
#include <compiler/graph.h>
#include <compiler/node.h>
#include <compiler/tensor.h>
#include <compiler/type.h>
#include <compiler/value.h>

namespace oniku {

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
        Dtype odtype = node->outputs()[i]->type().dtype();
        if (odtype == Dtype::kUnknown) {
            node->outputs()[i]->mutable_type()->set_dtype(dtype);
        } else {
            if (dtype != Dtype::kUnknown) CHECK_EQ(dtype, odtype) << "dtype mismatch for output #" << i << " of " << node->ToString();
        }
    };

    auto oset = [node, set](int i, Dtype dtype) {
        if (i < node->outputs().size()) set(i, dtype);
    };

    Dtype in0 = Dtype::kUnknown;
    if (node->inputs().size() >= 1) in0 = node->inputs()[0]->type().dtype();
    Dtype in1 = Dtype::kUnknown;
    if (node->inputs().size() >= 2) in1 = node->inputs()[1]->type().dtype();
    Dtype in2 = Dtype::kUnknown;
    if (node->inputs().size() >= 3) in2 = node->inputs()[2]->type().dtype();

    switch (node->op_type()) {
        case Node::kIdentity:
        case Node::kNeg:
        case Node::kAbs:
        case Node::kRelu:
        case Node::kFloor:
        case Node::kCeil:
        case Node::kSqueeze:
        case Node::kUnsqueeze:
        case Node::kFlatten:
        case Node::kSlice:
        case Node::kDynamicSlice:
        case Node::kReduceSum:
        case Node::kReduceSumSquare:
        case Node::kReduceL1:
        case Node::kReduceL2:
        case Node::kReduceLogSum:
        case Node::kReduceLogSumExp:
        case Node::kReduceMax:
        case Node::kReduceMin:
        case Node::kClip:
        case Node::kPad:
        case Node::kMaxPool:
        case Node::kGlobalMaxPool:
        case Node::kTranspose: {
            set(0, in0);
            break;
        }

        case Node::kReciprocal:
        case Node::kExp:
        case Node::kTanh:
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
        case Node::kOnikuxLinear: {
            set(0, coerce());
            break;
        }

        case Node::kNot:
        case Node::kEqual:
        case Node::kGreater:
        case Node::kLess: {
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

        case Node::kOneHot: {
            set(0, in2);
            break;
        }

        case Node::kReshape:
        case Node::kExpand:
        case Node::kOnikuxReduceSumTo: {
            CHECK(in1 == Dtype::kInt64 || in1 == Dtype::kUnknown) << in1.ToString() << " in " << node->ToString();
            set(0, in0);
            break;
        }

        case Node::kGather:
        case Node::kOnikuxSelectItem: {
            // TODO(hamaji): Need an update for the Python compiler.
            // CHECK(in1 == Dtype::kInt32 || in1 == Dtype::kInt64 || in1 == Dtype::kUnknown) << in1.ToString() << " in " <<
            // node->ToString();
            set(0, in0);
            break;
        }

        case Node::kOnikuxGatherGrad:
            // TODO(hamaji): Check for other inputs.
            set(0, in0);
            break;

        case Node::kRNN:
        case Node::kGRU:
        case Node::kLSTM: {
            Dtype dtype = CoerceDtype(in0, in1);
            if (node->inputs().size() >= 3) dtype = CoerceDtype(dtype, node->inputs()[2]->type().dtype());
            oset(0, dtype);
            oset(1, dtype);
            oset(2, dtype);
            break;
        }

        case Node::kOnikuxLSTMGrad: {
            // TODO(hamaji): Implement this.
            break;
        }

        case Node::kConv:
        case Node::kConvTranspose:
        case Node::kOnikuxConvGradWeight: {
            Dtype dtype = CoerceDtype(in0, in1);
            if (node->inputs().size() >= 3) dtype = CoerceDtype(dtype, node->inputs()[2]->type().dtype());
            oset(0, dtype);
            break;
        }

        case Node::kBatchNormalization: {
            Dtype dtype = coerce();
            set(0, dtype);
            for (int i = 1; i < 5; ++i) oset(i, dtype);
            break;
        }

        case Node::kOnikuxSoftmaxCrossEntropy: {
            // TODO(hamaji): Probably, better to fix the compiler.
            // CHECK(in1 == Dtype::kInt32 || in1 == Dtype::kInt64 || in1 == Dtype::kUnknown) << in1.ToString() << " in " <<
            // node->ToString();
            set(0, in0);
            break;
        }

        case Node::kOnikuxMaxPoolGrad:
        case Node::kOnikuxAveragePoolGrad:
        case Node::kOnikuxReluGrad:
        case Node::kOnikuxLRNGrad: {
            set(0, coerce());
            break;
        }

        case Node::kOnikuxBatchNormalizationGrad: {
            Dtype dtype = coerce();
            set(0, dtype);
            set(1, dtype);
            set(2, dtype);
            break;
        }

        case Node::kOnikuxConvTransposeWithDynamicOutputShape: {
            CHECK(in2 == Dtype::kInt64 || in2 == Dtype::kUnknown) << in1.ToString() << " in " << node->ToString();
            set(0, CoerceDtype(in0, in1));
            break;
        }

        case Node::kOnikuxSelectItemGrad: {
            // TODO(hamaji): Probably, better to fix the compiler.
            // CHECK(in1 == Dtype::kInt32 || in1 == Dtype::kInt64 || in1 == Dtype::kUnknown) << in1.ToString() << " in " <<
            // node->ToString();
            CHECK(in2 == Dtype::kInt64 || in2 == Dtype::kUnknown) << in2.ToString() << " in " << node->ToString();
            set(0, in0);
            break;
        }

        case Node::kIf: {
            // TODO(hamaji): Dtype inference for Loop/If is not implemented yet.
            break;
        }

        case Node::kLoop: {
            // TOOD(hamaji): Enable this check.
#if 0
            for (size_t i = 2; i < node->inputs().size(); ++i) {
                Value* input = node->inputs()[i];
                Value* output = node->outputs()[i - 2];
                CHECK_EQ(input->type().kind(), output->type().kind());
            }
            for (size_t i = 2; i < node->inputs().size(); ++i) {
                Value* input = node->inputs()[i];
                Value* body = node->body()->input_values()[i];
                CHECK_EQ(input->type().kind(), body->type().kind());
            }
            for (size_t i = 0; i < node->outputs().size(); ++i) {
                Value* output = node->inputs()[i];
                Value* body = node->body()->output_values()[i + 1];
                CHECK_EQ(output->type().kind(), body->type().kind());
            }
#endif

            // TODO(hamaji): Dtype inference for Loop/If is not implemented yet.
            break;
        }

        case Node::kOnikuxFusionGroup: {
            // TODO(hamaji): Dtype inference for Loop/If is not implemented yet.
            break;
        }

#if 0
        // TODO(hamaji): Revive Scan.
        case Node::kScan: {
            // TODO(hamaji): We assume when all inputs have the smae
            // dtypes, the outputs will be the same.
            Dtype dtype = in0;
            for (Value* in : node->inputs()) {
                if (in->type().dtype() != dtype) {
                    WARN_ONCE("Dtype inference for Scan with multiple types of dtypes");
                }
            }
            for (size_t i = 0; i < node->outputs().size(); ++i) {
                set(i, dtype);
            }
            break;
        }
#endif

        case Node::kSplit: {
            for (size_t i = 0; i < node->outputs().size(); ++i) {
                set(i, in0);
            }
            break;
        }

        case Node::kOnikuxPrint:
            break;

        case Node::kOnikuxNullConstant:
        case Node::kOnikuxDynamicSliceGrad:
        case Node::kOnikuxSequenceConstants:
        case Node::kOnikuxSequenceCreate:
        case Node::kOnikuxSequenceAppend:
        case Node::kOnikuxSequencePop:
        case Node::kOnikuxSequenceLookup:
        case Node::kOnikuxSequenceLookupGrad:
        case Node::kOnikuxSequenceStack:
        case Node::kOnikuxSequenceConcat:
        case Node::kOnikuxSequenceSplitAxis:
        case Node::kOnikuxSequenceSeparate:
        case Node::kOnikuxSequenceUnpad:
        case Node::kOnikuxSequenceLengths:
        case Node::kOnikuxSequenceSize:
        case Node::kOnikuxSequencePad:
        case Node::kOnikuxSequenceRange:
        case Node::kOnikuxSequenceGetSlice:
        case Node::kOnikuxSequenceGetSliceGrad:
        case Node::kOnikuxGenericLen:
        case Node::kOnikuxGenericGetItem:
        case Node::kOnikuxGenericGetSlice:
        case Node::kOnikuxGenericAdd:
        case Node::kOnikuxGenericAccumulateGrad:
        case Node::kOnikuxGenericIs: {
            // TODO(hamaji): Consider implementing dtype inference for sequences.
            break;
        }
    }
}

}  // namespace oniku
