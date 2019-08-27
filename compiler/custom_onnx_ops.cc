#include "compiler/onnx.h"
#include "onnx/defs/schema.h"

#define ONNX_CHAINER_OPERATOR_SET_SCHEMA(name, ver, impl) \
    ONNX_OPERATOR_SET_SCHEMA_EX(name, Chainer, chainer_compiler::CHAINER_ONNX_DOMAIN, ver, false, impl)
#define ONNX_WORKAROUND_OPERATOR_SET_SCHEMA(name, ver, impl) ONNX_OPERATOR_SET_SCHEMA_EX(name, Onnx, ONNX_DOMAIN, ver, false, impl)

namespace ONNX_NAMESPACE {

namespace {

void InferLinear(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    int n_batch_axes = getAttribute(ctx, "n_batch_axes", 1);
    auto& first_input_shape = getInputShape(ctx, 0);
    auto& second_input_shape = getInputShape(ctx, 1);

    if (n_batch_axes > first_input_shape.dim_size()) {
        return;
    }
    if (1 > second_input_shape.dim_size()) {
        return;
    }

    auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < n_batch_axes; ++i) {
        output_shape->add_dim()->CopyFrom(first_input_shape.dim(i));
    }
    output_shape->add_dim()->CopyFrom(second_input_shape.dim(0));
}

}  // namespace

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerLinear,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Input(1, "W", "Weight tensor", "T")
                .Input(2, "B", "Bias tensor", "T", OpSchema::Optional)
                .Output(0, "Y", "Output tensor", "T")
                .Attr("n_batch_axes", "The number of batch axes.", AttributeProto::INT, static_cast<int64_t>(1))
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeAndShapeInferenceFunction(InferLinear));

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerSoftmaxCrossEntropy,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Input(1, "T", "Target labels", "I")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeConstraint("I", {"tensor(int32)", "tensor(int64)"}, "Constrain input and output types to signed numeric tensors.")
                .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
                    propagateElemTypeFromInputToOutput(ctx, 0, 0);
                    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
                }));

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerSelectItem,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Input(1, "indices", "Tensors of int32/int64 indices", "I")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeConstraint("I", {"tensor(int32)", "tensor(int64)"}, "Constrain input and output types to signed numeric tensors.")
                .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
                    propagateElemTypeFromInputToOutput(ctx, 0, 0);
                    if (!hasNInputShapes(ctx, 2)) {
                        return;
                    }
                    const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
                    if (data_shape.dim_size() != 2) {
                        fail_shape_inference("data tensor must be rank == 2");
                    }
                    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->CopyFrom(data_shape.dim(0));
                }));

namespace {

void InferROI(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    std::vector<int64_t> output_shape;
    if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
        if (output_shape.size() != 2) {
            fail_shape_inference("Attribute output_shape has incorrect size");
        }
    } else {
        fail_shape_inference("Attribute output_shape must be specified");
    }

    auto output = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    auto add_dim = [&output](const TypeProto& type, int index) {
        auto new_dim = output->add_dim();
        if (!type.has_tensor_type() || !type.tensor_type().has_shape() || index >= type.tensor_type().shape().dim().size()) {
            return;
        }
        new_dim->CopyFrom(type.tensor_type().shape().dim(index));
    };

    add_dim(*ctx.getInputType(1), 0);
    add_dim(*ctx.getInputType(0), 1);
    output->add_dim()->set_dim_value(output_shape[0]);
    output->add_dim()->set_dim_value(output_shape[1]);
}

}  // namespace

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerROIAveragePool2D,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Input(1, "rois", "Input tensor", "T")
                .Input(2, "roi_indices", "Input tensor", "I")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
                .TypeAndShapeInferenceFunction(InferROI));

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerROIMaxPool2D,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Input(1, "rois", "Input tensor", "T")
                .Input(2, "roi_indices", "Input tensor", "I")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
                .TypeAndShapeInferenceFunction(InferROI));

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerROIAverageAlign2D,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Input(1, "rois", "Input tensor", "T")
                .Input(2, "roi_indices", "Input tensor", "I")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
                .TypeAndShapeInferenceFunction(InferROI));

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerROIMaxAlign2D,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Input(1, "rois", "Input tensor", "T")
                .Input(2, "roi_indices", "Input tensor", "I")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
                .TypeAndShapeInferenceFunction(InferROI));

namespace {

void InferResizeImages(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    std::vector<int64_t> output_shape;
    if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
        if (output_shape.size() != 2) {
            fail_shape_inference("Attribute output_shape has incorrect size");
        }
    } else {
        fail_shape_inference("Attribute output_shape must be specified");
    }

    auto output = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    auto add_dim = [&output](const TypeProto& type, int index) {
        auto new_dim = output->add_dim();
        if (!type.has_tensor_type() || !type.tensor_type().has_shape() || index >= type.tensor_type().shape().dim().size()) {
            return;
        }
        new_dim->CopyFrom(type.tensor_type().shape().dim(index));
    };
    add_dim(*ctx.getInputType(0), 0);
    add_dim(*ctx.getInputType(0), 1);
    output->add_dim()->set_dim_value(output_shape[0]);
    output->add_dim()->set_dim_value(output_shape[1]);
}

}  // namespace

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerResizeImages,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .Attr("output_shape", "The size of the output image.", AttributeProto::INTS)
                .TypeAndShapeInferenceFunction(InferResizeImages));

// TODO(hamaji): Remove this once
// https://github.com/onnx/onnx/pull/1855 is merged.
static const char* Expand_ver9_doc = R"DOC(
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimension must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
)DOC";
ONNX_WORKAROUND_OPERATOR_SET_SCHEMA(
        Expand,
        9,
        OpSchema()
                .SetDoc(Expand_ver9_doc)
                .Input(0, "input", "Input tensor", "T")
                .Input(1, "shape", "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule", "tensor(int64)")
                .Output(0, "output", "Output tensor", "T")
                .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensors.")
                .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
                    // Type inference
                    propagateElemTypeFromInputToOutput(ctx, 0, 0);
                    // Shape Inference if 2nd input data (the target shape) is available
                    const TensorProto* target_shape_initializer = ctx.getInputData(1);
                    if (!target_shape_initializer) {
                        return;
                    }
                    // The target_shape vector represents the specified shape for output.
                    std::vector<int64_t> target_shape;
                    if (target_shape_initializer->has_raw_data()) {
                        const std::string& bytes = target_shape_initializer->raw_data();
                        target_shape.insert(
                                target_shape.end(),
                                reinterpret_cast<const int64_t*>(bytes.c_str()),
                                reinterpret_cast<const int64_t*>(bytes.c_str() + bytes.size()));
                    } else {
                        const auto& data = target_shape_initializer->int64_data();
                        target_shape.insert(target_shape.end(), data.begin(), data.end());
                    }
                    auto& input_shape = getInputShape(ctx, 0);
                    auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
                    int num_target_dims = static_cast<int>(target_shape.size());
                    int num_new_dims = std::max(input_shape.dim_size(), num_target_dims);
                    for (int i = 0; i < num_new_dims; ++i) {
                        auto* new_dim = output_shape->add_dim();
                        int target_index = i + num_target_dims - num_new_dims;
                        int64_t target_dim = target_index < 0 ? 1 : target_shape[target_index];
                        int input_index = i + input_shape.dim_size() - num_new_dims;
                        if (input_index < 0) {
                            new_dim->set_dim_value(target_dim);
                        } else if (input_shape.dim(input_index).has_dim_value()) {
                            const int64_t input_dim = input_shape.dim(input_index).dim_value();
                            if (input_dim != target_dim && input_dim != 1) {
                                if (target_dim != 1) {
                                    fail_shape_inference("Incompatible dimensions in Expand (", input_dim, " vs ", target_dim);
                                }
                                target_dim = input_dim;
                            }
                            new_dim->set_dim_value(target_dim);
                        } else if (target_dim != 1) {
                            new_dim->set_dim_value(target_dim);
                        } else {
                            new_dim->CopyFrom(input_shape.dim(input_index));
                        }
                    }
                }));

namespace {

void InferPadBatchSize(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    const int64_t batch_size = getAttribute(ctx, "size", -1);
    if (batch_size < 1) {
        fail_shape_inference("invalid batch size");
    }
    const TypeProto& type = *ctx.getInputType(0);
    if (!type.has_tensor_type() || !type.tensor_type().has_shape() || type.tensor_type().shape().dim_size() < 1) {
        return;
    }

    auto output = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    output->add_dim()->set_dim_value(batch_size);
    for (int i = 1; i < type.tensor_type().shape().dim_size(); ++i) {
        output->add_dim()->CopyFrom(ctx.getInputType(0)->tensor_type().shape().dim(i));
    }
}

}  // namespace

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerPadBatchSize,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Output(0, "Y", "Output tensor", "T")
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .Attr("size", "The batch size of the output.", AttributeProto::INT)
                .TypeAndShapeInferenceFunction(InferPadBatchSize));

namespace {

static const char* Split_ver9_doc =
        R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
)DOC";

void InferSplit(InferenceContext& ctx) {
    for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
        propagateElemTypeFromInputToOutput(ctx, 0, i);
    }

    if (!hasNInputShapes(ctx, 1)) {
        return;
    }

    auto axisAttr = ctx.getAttribute("axis");
    int axis = axisAttr ? static_cast<int>(axisAttr->i()) : 0;
    if (axis < 0) {
        return;
    }
    if (!ctx.getInputType(0)->tensor_type().has_shape()) {
        return;
    }
    const auto& shape = ctx.getInputType(0)->tensor_type().shape();
    if (axis >= shape.dim_size()) {
        fail_type_inference("Invalid value of attribute 'axis'");
    }
    const auto& splitDim = shape.dim(axis);
    if (!splitDim.has_dim_value()) {
        return;
    }
    int splitDimValue = static_cast<int>(splitDim.dim_value());

    std::vector<int64_t> split;
    if (getRepeatedAttribute(ctx, "split", split)) {
        if (split.size() != ctx.getNumOutputs()) {
            return;
        }
        int64_t totalDim = 0;
        for (int64_t d : split) totalDim += d;
        if (totalDim != splitDimValue) {
            return;
        }
    } else {
        int chunkSize = splitDimValue / static_cast<int>(ctx.getNumOutputs());
        int leftOver = splitDimValue - (chunkSize * static_cast<int>(ctx.getNumOutputs()));
        for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
            split.push_back(i < leftOver ? chunkSize + 1 : chunkSize);
        }
    }

    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
        *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
        ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
    }
}

}  // namespace

ONNX_WORKAROUND_OPERATOR_SET_SCHEMA(
        Split,
        9,
        OpSchema()
                .Input(0, "input", "The tensor to split", "T")
                .Output(0, "outputs", "One or more outputs forming list of tensors after splitting", "T", OpSchema::Variadic)
                .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
                .Attr("axis", "Which axis to split on.", AttributeProto::INT, static_cast<int64_t>(0))
                .Attr("split", "length of each output", AttributeProto::INTS, OPTIONAL)
                .SetDoc(Split_ver9_doc)
                .TypeAndShapeInferenceFunction(InferSplit));

namespace {

void InferReduceSumTo(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    const TensorProto* targetShapeInitializer = ctx.getInputData(1);
    if (!targetShapeInitializer) {
        return;
    }
    std::vector<int64_t> targetShape;
    if (targetShapeInitializer->has_raw_data()) {
        const std::string& bytes = targetShapeInitializer->raw_data();
        targetShape.insert(
                targetShape.end(),
                reinterpret_cast<const int64_t*>(bytes.c_str()),
                reinterpret_cast<const int64_t*>(bytes.c_str() + bytes.size()));
    } else {
        const auto& data = targetShapeInitializer->int64_data();
        targetShape.insert(targetShape.end(), data.begin(), data.end());
    }
    auto* outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < static_cast<int>(targetShape.size()); ++i) {
        outputShape->add_dim()->set_dim_value(targetShape[i]);
    }
}

}  // namespace

ONNX_CHAINER_OPERATOR_SET_SCHEMA(
        ChainerReduceSumTo,
        9,
        OpSchema()
                .Input(0, "input", "An input tensor", "T")
                .Input(1, "shape", "An output shape", "T")
                .Output(0, "output", "Reduced output tensor", "T")
                .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
                .TypeAndShapeInferenceFunction(InferReduceSumTo));

class Custom_OpSet_Onnx_ver9 {
public:
    static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
        ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(chainer_compiler::CHAINER_ONNX_DOMAIN, 9, 9);
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerPadBatchSize)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerReduceSumTo)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerROIAverageAlign2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerROIAveragePool2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerROIMaxAlign2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerROIMaxPool2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerLinear)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerResizeImages)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerSoftmaxCrossEntropy)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Chainer, 9, ChainerSelectItem)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Expand)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Split)>());
    }
};

}  // namespace ONNX_NAMESPACE

namespace chainer_compiler {

namespace {

bool RegisterCustomOnnxOperatorSetSchemaImpl() {
    ONNX_NAMESPACE::RegisterOpSetSchema<ONNX_NAMESPACE::Custom_OpSet_Onnx_ver9>();
    return true;
}

}  // namespace

void RegisterCustomOnnxOperatorSetSchema() {
    // Run just once.
    static bool unused = RegisterCustomOnnxOperatorSetSchemaImpl();
    (void)unused;
}

}  // namespace chainer_compiler
