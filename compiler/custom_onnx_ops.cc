#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    OnikuxSoftmaxCrossEntropy,
    9,
    OpSchema()
        .SetDoc("TBD")
        .Input(0, "X", "Input tensor", "T")
        .Input(0, "T", "Target labels", "I")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(float16)",
             "tensor(double)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeConstraint(
            "I",
            {"tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
            ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape();
            }));

class Custom_OpSet_Onnx_ver9 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, OnikuxSoftmaxCrossEntropy)>());
  }
};

}  // namespace ONNX_NAMESPACE

namespace oniku {

void RegisterCustomOnnxOperatorSetSchema() {
    ONNX_NAMESPACE::RegisterOpSetSchema<ONNX_NAMESPACE::Custom_OpSet_Onnx_ver9>();
}

}  // namespace oniku
