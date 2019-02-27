#include "compiler/xcvm/config.h"

#include <set>

#include <common/log.h>
#include <compiler/config.h>

namespace chainer_compiler {
namespace xcvm {
namespace {

class XCVMCompilerConfig : public CompilerConfig {
public:
    explicit XCVMCompilerConfig(bool diversed) {
        name_ = diversed ? "xcvm_test" : "xcvm";

        CHECK(op_set_.emplace(Node::kAbs).second);
        CHECK(op_set_.emplace(Node::kAdd).second);
        CHECK(op_set_.emplace(Node::kAnd).second);
        CHECK(op_set_.emplace(Node::kArgMax).second);
        CHECK(op_set_.emplace(Node::kAveragePool).second);
        CHECK(op_set_.emplace(Node::kBatchNormalization).second);
        CHECK(op_set_.emplace(Node::kCast).second);
        CHECK(op_set_.emplace(Node::kCeil).second);
        CHECK(op_set_.emplace(Node::kClip).second);
        CHECK(op_set_.emplace(Node::kConcat).second);
        CHECK(op_set_.emplace(Node::kConstant).second);
        CHECK(op_set_.emplace(Node::kConstantFill).second);
        CHECK(op_set_.emplace(Node::kConv).second);
        CHECK(op_set_.emplace(Node::kConvTranspose).second);
        CHECK(op_set_.emplace(Node::kDiv).second);
        CHECK(op_set_.emplace(Node::kDropout).second);
        CHECK(op_set_.emplace(Node::kDynamicSlice).second);
        CHECK(op_set_.emplace(Node::kElu).second);
        CHECK(op_set_.emplace(Node::kEqual).second);
        CHECK(op_set_.emplace(Node::kExp).second);
        CHECK(op_set_.emplace(Node::kExpand).second);
        CHECK(op_set_.emplace(Node::kFloor).second);
        CHECK(op_set_.emplace(Node::kGRU).second);
        CHECK(op_set_.emplace(Node::kGather).second);
        CHECK(op_set_.emplace(Node::kGemm).second);
        CHECK(op_set_.emplace(Node::kGreater).second);
        CHECK(op_set_.emplace(Node::kHardmax).second);
        CHECK(op_set_.emplace(Node::kIdentity).second);
        CHECK(op_set_.emplace(Node::kIf).second);
        CHECK(op_set_.emplace(Node::kLRN).second);
        CHECK(op_set_.emplace(Node::kLSTM).second);
        CHECK(op_set_.emplace(Node::kLeakyRelu).second);
        CHECK(op_set_.emplace(Node::kLog).second);
        CHECK(op_set_.emplace(Node::kLogSoftmax).second);
        CHECK(op_set_.emplace(Node::kLoop).second);
        CHECK(op_set_.emplace(Node::kMatMul).second);
        CHECK(op_set_.emplace(Node::kMax).second);
        CHECK(op_set_.emplace(Node::kMaxPool).second);
        CHECK(op_set_.emplace(Node::kMul).second);
        CHECK(op_set_.emplace(Node::kNeg).second);
        CHECK(op_set_.emplace(Node::kNot).second);
        CHECK(op_set_.emplace(Node::kOneHot).second);
        CHECK(op_set_.emplace(Node::kChainerAveragePoolGrad).second);
        CHECK(op_set_.emplace(Node::kChainerAveragePoolGradNoCtx).second);
        CHECK(op_set_.emplace(Node::kChainerBatchNormalizationGrad).second);
        CHECK(op_set_.emplace(Node::kChainerConvGradWeight).second);
        CHECK(op_set_.emplace(Node::kChainerConvTransposeWithDynamicOutputShape).second);
        CHECK(op_set_.emplace(Node::kChainerDoSomething).second);
        CHECK(op_set_.emplace(Node::kChainerDynamicSliceGrad).second);
        CHECK(op_set_.emplace(Node::kChainerFusionGroup).second);
        CHECK(op_set_.emplace(Node::kChainerGatherGrad).second);
        CHECK(op_set_.emplace(Node::kChainerGenericAccumulateGrad).second);
        CHECK(op_set_.emplace(Node::kChainerGenericAdd).second);
        CHECK(op_set_.emplace(Node::kChainerGenericGetItem).second);
        CHECK(op_set_.emplace(Node::kChainerGenericGetSlice).second);
        CHECK(op_set_.emplace(Node::kChainerGenericIs).second);
        CHECK(op_set_.emplace(Node::kChainerGenericLen).second);
        CHECK(op_set_.emplace(Node::kChainerGetItem).second);
        CHECK(op_set_.emplace(Node::kChainerGetItemGrad).second);
        CHECK(op_set_.emplace(Node::kChainerLRNGrad).second);
        CHECK(op_set_.emplace(Node::kChainerLSTMGrad).second);
        CHECK(op_set_.emplace(Node::kChainerMaxPoolGrad).second);
        CHECK(op_set_.emplace(Node::kChainerMaxPoolGradNoCtx).second);
        CHECK(op_set_.emplace(Node::kChainerNullConstant).second);
        CHECK(op_set_.emplace(Node::kChainerPrint).second);
        CHECK(op_set_.emplace(Node::kChainerReduceSumTo).second);
        CHECK(op_set_.emplace(Node::kChainerReluGrad).second);
        CHECK(op_set_.emplace(Node::kChainerROIAverageAlign2D).second);
        CHECK(op_set_.emplace(Node::kChainerROIAveragePool2D).second);
        CHECK(op_set_.emplace(Node::kChainerROIMaxAlign2D).second);
        CHECK(op_set_.emplace(Node::kChainerROIMaxPool2D).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceAppend).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceConcat).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceConstants).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceCreate).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceGetSlice).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceGetSliceGrad).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceLengths).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceLookup).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceLookupGrad).second);
        CHECK(op_set_.emplace(Node::kChainerSequencePad).second);
        CHECK(op_set_.emplace(Node::kChainerSequencePop).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceRange).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceSeparate).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceSize).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceSplitAxis).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceStack).second);
        CHECK(op_set_.emplace(Node::kChainerSequenceUnpad).second);
        CHECK(op_set_.emplace(Node::kDepthToSpace).second);
        CHECK(op_set_.emplace(Node::kEyeLike).second);
        CHECK(op_set_.emplace(Node::kOr).second);
        CHECK(op_set_.emplace(Node::kPad).second);
        CHECK(op_set_.emplace(Node::kPow).second);
        CHECK(op_set_.emplace(Node::kRNN).second);
        CHECK(op_set_.emplace(Node::kReciprocal).second);
        CHECK(op_set_.emplace(Node::kReduceMax).second);
        CHECK(op_set_.emplace(Node::kReduceMean).second);
        CHECK(op_set_.emplace(Node::kReduceSum).second);
        CHECK(op_set_.emplace(Node::kReduceSumSquare).second);
        CHECK(op_set_.emplace(Node::kRelu).second);
        CHECK(op_set_.emplace(Node::kReshape).second);
        CHECK(op_set_.emplace(Node::kSelu).second);
        CHECK(op_set_.emplace(Node::kShape).second);
        CHECK(op_set_.emplace(Node::kSigmoid).second);
        CHECK(op_set_.emplace(Node::kSize).second);
        CHECK(op_set_.emplace(Node::kSlice).second);
        CHECK(op_set_.emplace(Node::kSoftmax).second);
        CHECK(op_set_.emplace(Node::kSpaceToDepth).second);
        CHECK(op_set_.emplace(Node::kSplit).second);
        CHECK(op_set_.emplace(Node::kSqrt).second);
        CHECK(op_set_.emplace(Node::kSqueeze).second);
        CHECK(op_set_.emplace(Node::kSub).second);
        CHECK(op_set_.emplace(Node::kTanh).second);
        CHECK(op_set_.emplace(Node::kTranspose).second);
        CHECK(op_set_.emplace(Node::kUnsqueeze).second);
        CHECK(op_set_.emplace(Node::kUpsample).second);
        CHECK(op_set_.emplace(Node::kXor).second);

        if (diversed) {
            // SelectItem seemed to be slow on GPU.
            CHECK(op_set_.emplace(Node::kChainerSelectItem).second);
            CHECK(op_set_.emplace(Node::kChainerSelectItemGrad).second);
        } else {
            CHECK(op_set_.emplace(Node::kChainerLinear).second);
            CHECK(op_set_.emplace(Node::kChainerLinearGradWeight).second);
        }
    }

    virtual ~XCVMCompilerConfig() = default;

    virtual bool HasOp(Node::OpType op) const {
        return op_set_.count(op);
    }

    virtual std::string name() const {
        return name_;
    }

protected:
    std::string name_;
    std::set<Node::OpType> op_set_;
};

}  // namespace

std::unique_ptr<CompilerConfig> GetCompilerConfig(bool diversed) {
    return std::unique_ptr<CompilerConfig>(new XCVMCompilerConfig(diversed));
}

}  // namespace xcvm
}  // namespace chainer_compiler
