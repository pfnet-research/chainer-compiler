#include <compiler/xcvm/config.h>

#include <set>

#include <common/log.h>
#include <compiler/config.h>

namespace oniku {
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
        CHECK(op_set_.emplace(Node::kOnikuxAveragePoolGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxBatchNormalizationGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxConvGradWeight).second);
        CHECK(op_set_.emplace(Node::kOnikuxConvTransposeWithDynamicOutputShape).second);
        CHECK(op_set_.emplace(Node::kOnikuxDynamicSliceGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxFusionGroup).second);
        CHECK(op_set_.emplace(Node::kOnikuxGatherGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxGenericAccumulateGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxGenericAdd).second);
        CHECK(op_set_.emplace(Node::kOnikuxGenericGetItem).second);
        CHECK(op_set_.emplace(Node::kOnikuxGenericGetSlice).second);
        CHECK(op_set_.emplace(Node::kOnikuxGenericIs).second);
        CHECK(op_set_.emplace(Node::kOnikuxGenericLen).second);
        CHECK(op_set_.emplace(Node::kOnikuxLRNGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxLSTMGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxMaxPoolGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxNullConstant).second);
        CHECK(op_set_.emplace(Node::kOnikuxPrint).second);
        CHECK(op_set_.emplace(Node::kOnikuxReduceSumTo).second);
        CHECK(op_set_.emplace(Node::kOnikuxReluGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceAppend).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceConcat).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceConstants).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceCreate).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceGetSlice).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceGetSliceGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceLengths).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceLookup).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceLookupGrad).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequencePad).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequencePop).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceRange).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceSeparate).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceSize).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceSplitAxis).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceStack).second);
        CHECK(op_set_.emplace(Node::kOnikuxSequenceUnpad).second);
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
        CHECK(op_set_.emplace(Node::kSplit).second);
        CHECK(op_set_.emplace(Node::kSqrt).second);
        CHECK(op_set_.emplace(Node::kSqueeze).second);
        CHECK(op_set_.emplace(Node::kSub).second);
        CHECK(op_set_.emplace(Node::kTanh).second);
        CHECK(op_set_.emplace(Node::kTranspose).second);
        CHECK(op_set_.emplace(Node::kUnsqueeze).second);
        CHECK(op_set_.emplace(Node::kXor).second);

        if (diversed) {
            // SelectItem seemed to be slow on GPU.
            CHECK(op_set_.emplace(Node::kOnikuxSelectItem).second);
            CHECK(op_set_.emplace(Node::kOnikuxSelectItemGrad).second);
        } else {
            CHECK(op_set_.emplace(Node::kOnikuxLinear).second);
            CHECK(op_set_.emplace(Node::kOnikuxLinearGradWeight).second);
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
}  // namespace oniku
