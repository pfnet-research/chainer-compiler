#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
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
                .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
                    propagateElemTypeFromInputToOutput(ctx, 0, 0);
                    auto& first_input_shape = getInputShape(ctx, 0);
                    auto& second_input_shape = getInputShape(ctx, 1);
                    updateOutputShape(ctx, 0, {first_input_shape.dim(0), second_input_shape.dim(0)});
                }));

ONNX_OPERATOR_SET_SCHEMA(
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

ONNX_OPERATOR_SET_SCHEMA(
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

// From onnx/onnx/defs/nn/defs.cc
void convPoolTypeAndShapeInference(InferenceContext& ctx, bool use_dilation, bool require_kernel_shape) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (ctx.getNumOutputs() > 1) {
        // MaxPool with two outputs case.
        auto output_type = ctx.getOutputType(1);
        if (output_type->value_case() == TypeProto::kTensorType || output_type->value_case() == TypeProto::VALUE_NOT_SET) {
            output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
        }
    }

    // we need the first input shape for this inference.
    if (!hasNInputShapes(ctx, 1)) {
        return;
    }

    // if kernel shape is an input (and not attribute)
    // we need the shape of the second input.
    if (!require_kernel_shape && !hasNInputShapes(ctx, 2)) {
        return;
    }

    // don't bother with legacy auto_pad for now
    if (ctx.getAttribute("auto_pad")) {
        return;
    }

    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    if (input_shape.dim_size() < 2) {
        fail_shape_inference("Input tensor must have atleast 2 dimensions");
    }

    // first dim is the batch axis and the next is the number of channels.
    size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

    // Pooling operations don't support dilation, only Conv. For
    // simplicity of the code, we just treat them as having all-1s
    // dilation.
    std::vector<int64_t> dilations;
    if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
        if (dilations.size() != n_input_dims) {
            fail_shape_inference("Attribute dilations has incorrect size");
        }
    } else {
        dilations.assign(n_input_dims, 1);
    }

    std::vector<int64_t> pads;
    if (getRepeatedAttribute(ctx, "pads", pads)) {
        if (pads.size() != n_input_dims * 2) {
            fail_shape_inference("Attribute pads has incorrect size");
        }
    } else {
        pads.assign(n_input_dims * 2, 0);
    }

    std::vector<int64_t> strides;
    if (getRepeatedAttribute(ctx, "strides", strides)) {
        if (strides.size() != n_input_dims) {
            fail_shape_inference("Attribute strides has incorrect size");
        }
    } else {
        strides.assign(n_input_dims, 1);
    }

    std::vector<int64_t> kernel_shape;
    if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
        if (kernel_shape.size() != n_input_dims) {
            fail_shape_inference("Attribute kernel_shape has incorrect size");
        }
    } else if (require_kernel_shape) {
        fail_shape_inference("Attribute kernel_shape must be specified");
    } else {
        auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
        for (int i = 2; i < second_input_shape.dim_size(); ++i) {
            if (!second_input_shape.dim(i).has_dim_value()) {
                return;
            }
            kernel_shape.push_back(second_input_shape.dim(i).dim_value());
        }
    }

    auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

    if (require_kernel_shape) {
        // add the first two dimensions from the input.
        *output_shape->add_dim() = input_shape.dim(0);
        *output_shape->add_dim() = input_shape.dim(1);
    } else {
        *output_shape->add_dim() = input_shape.dim(0);
        auto& second_input_shape = getInputShape(ctx, 1);
        if (second_input_shape.dim_size() < 1) {
            fail_shape_inference("Second input tensor has wrong dimension");
        }
        *output_shape->add_dim() = second_input_shape.dim(0);
    }

    // EDIT(hamaji): Check if `chainer_cover_all` is set.
    const bool cover_all = getAttribute(ctx, "chainer_cover_all", 0);

    int kernel_shape_size = static_cast<int>(kernel_shape.size());
    for (int i = 0; i < kernel_shape_size; ++i) {
        auto newdim = output_shape->add_dim();
        if (!input_shape.dim(2 + i).has_dim_value()) {
            continue;
        }
        // how big is the input, including padding
        int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
        effective_input_size += pads[i];
        effective_input_size += pads[i + kernel_shape_size];

        int64_t effective_kernel_size = kernel_shape[i];
        // accounting for dilation, how big is the kernel in this dimension
        effective_kernel_size = (effective_kernel_size - 1) * dilations[i] + 1;

        // how many times we can move the kernel from it's initial position, based
        // on the stride
        int64_t strided_kernel_positions = (effective_input_size - effective_kernel_size) / strides[i];
        // EDIT(hamaji): Adjustment for `chainer_cover_all`.
        if (cover_all && (effective_input_size - effective_kernel_size) % strides[i]) {
            ++strided_kernel_positions;
        }

        // add in the initial position
        newdim->set_dim_value(1 + strided_kernel_positions);
    }

    if (ctx.getNumOutputs() > 1) {
        // MaxPool with two outputs case.
        auto second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
        second_output_shape->CopyFrom(*output_shape);
    }
}

}  // namespace

ONNX_OPERATOR_SET_SCHEMA(
        MaxPool,
        9,
        OpSchema()
                .SetDoc("TBD")
                .Input(0, "X", "Input tensor", "T")
                .Output(0, "Y", "Output tensor", "T")
                .Output(1, "Indices", "Indices tensor", "I", OpSchema::Optional)
                .TypeConstraint(
                        "T",
                        {"tensor(float)", "tensor(float16)", "tensor(double)"},
                        "Constrain input and output types to signed numeric tensors.")
                .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64")
                .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { convPoolTypeAndShapeInference(ctx, false, true); }));

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
    output->add_dim()->CopyFrom(ctx.getInputType(1)->tensor_type().shape().dim(0));
    output->add_dim()->CopyFrom(ctx.getInputType(0)->tensor_type().shape().dim(1));
    output->add_dim()->set_dim_value(output_shape[0]);
    output->add_dim()->set_dim_value(output_shape[1]);
}

}  // namespace

ONNX_OPERATOR_SET_SCHEMA(
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

ONNX_OPERATOR_SET_SCHEMA(
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

ONNX_OPERATOR_SET_SCHEMA(
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

ONNX_OPERATOR_SET_SCHEMA(
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
ONNX_OPERATOR_SET_SCHEMA(
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

class Custom_OpSet_Onnx_ver9 {
public:
    static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ChainerROIAverageAlign2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ChainerROIAveragePool2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ChainerROIMaxAlign2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ChainerROIMaxPool2D)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ChainerLinear)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ChainerSoftmaxCrossEntropy)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ChainerSelectItem)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Expand)>());
        fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MaxPool)>());
    }
};

}  // namespace ONNX_NAMESPACE

namespace chainer_compiler {

void RegisterCustomOnnxOperatorSetSchema() {
    ONNX_NAMESPACE::RegisterOpSetSchema<ONNX_NAMESPACE::Custom_OpSet_Onnx_ver9>();
}

}  // namespace chainer_compiler
