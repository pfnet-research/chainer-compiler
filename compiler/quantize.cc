#include <compiler/quantize.h>

#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/statistics.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/value.h>
#include <runtime/chainerx_util.h>

namespace chainer_compiler {

namespace {

struct QuantizedInput {
    Value* input;
    Value* scale;
    Value* zero_point;
};

struct QuantizedOutput {
    Value* output;
    Value* scale;
    Value* zero_point;
    chainerx::Shape scale_shape;
    chainerx::Shape zero_point_shape;
};

enum class DataMode {
    Linear_NonScaled = 0,
    Linear_Scaled = 1,
};

std::ostream& operator<<(std::ostream& os, DataMode mode) {
    switch (mode) {
        case DataMode::Linear_NonScaled:
            os << "Linear_NonScaled";
            break;
        case DataMode::Linear_Scaled:
            os << "Linear_Scaled";
            break;
        default:
            os << "Unknown: " << int(mode);
    }
    return os;
}

DataMode ModeForDataType(Dtype dtype) {
    return dtype == Dtype::kInt8 ? DataMode::Linear_Scaled : DataMode::Linear_NonScaled;
}

float QRangeForQtype(Dtype dtype) {
    return dtype == Dtype::kUInt8 ? 255 : 254;
}

struct QuantizedData {
    float rmin, rmax, scale;
    int64_t zero_point;
    chainerx::Array data;
};

QuantizedData QuantizeData(const QuantizationOptions& opts, const chainerx::Array& data, float quantize_range, DataMode mode) {
    float rmin = std::min(float(chainerx::AsScalar(data.Min())), 0.f);
    float rmax = std::max(float(chainerx::AsScalar(data.Max())), 0.f);

    float scale = 0.f;
    int64_t zero_point = 0;
    chainerx::Array quantized;
    if (mode == DataMode::Linear_Scaled) {
        float max_range = std::max(std::abs(rmin), std::abs(rmax));
        scale = max_range * 2 / quantize_range;
        zero_point = 0;
        quantized = runtime::SlowRound(data / scale).AsType(chainerx::Dtype::kInt8);
    } else {
        CHECK_EQ(mode, DataMode::Linear_NonScaled);
        scale = rmin != rmax ? (rmax - rmin) / quantize_range : 1.f;
        zero_point = std::rint((0 - rmin) / scale);
        quantized = runtime::SlowRound(data / scale).AsType(chainerx::Dtype::kUInt8);
    }
    return {rmin, rmax, scale, zero_point, quantized};
}

QuantizedInput QuantizeWeight(const QuantizationOptions& opts, GraphBuilder* gb, const chainerx::Array& w, Dtype dtype) {
    QuantizedData quantized = QuantizeData(opts, w, QRangeForQtype(dtype), ModeForDataType(dtype));
    Value* scale = gb->Const(chainerx::Full({}, quantized.scale, chainerx::Dtype::kFloat32, w.device()));
    Value* zero_point = gb->Const(chainerx::Full({}, quantized.zero_point, chainerx::Dtype::kFloat32, w.device()));
    return {gb->Const(quantized.data), scale, zero_point};
}

QuantizedInput QuantizeWeightConvolution(const QuantizationOptions& opts, GraphBuilder* gb, const chainerx::Array& w, Dtype dtype) {
    if (opts.per_channel) {
        return QuantizeWeight(opts, gb, w, dtype);
    }

    int64_t channel_count = w.shape()[0];
    std::vector<float> rmin_list, rmax_list, scale_list;
    std::vector<uint8_t> zero_point_list;
    std::vector<QuantizedData> quantized_per_channel_data_list;
    std::vector<chainerx::Array> quantized_weights;

    for (int64_t i = 0; i < channel_count; ++i) {
        chainerx::Array per_channel_data = w.At({i});
        QuantizedData quantized = QuantizeData(opts, per_channel_data, QRangeForQtype(dtype), ModeForDataType(dtype));
        rmin_list.push_back(quantized.rmin);
        rmax_list.push_back(quantized.rmax);
        zero_point_list.push_back(quantized.zero_point);
        scale_list.push_back(quantized.scale);
        quantized_weights.push_back(quantized.data);
        quantized_per_channel_data_list.push_back(quantized);
    }

    chainerx::Shape rehape_dims = w.shape();
    Value* scale = gb->Const(runtime::MakeArray(chainerx::Dtype::kFloat32, {channel_count}, scale_list.data()));
    Value* zero_point = gb->Const(runtime::MakeArray(dtype.chx(), {channel_count}, zero_point_list.data()));
    return {gb->Const(chainerx::Stack(quantized_weights)), scale, zero_point};
}

QuantizedOutput QuantizeOutput(const QuantizationOptions& opts, GraphBuilder* gb, Value* output) {
    auto it = opts.output_quantization_params.find(output->name());
    CHECK(it != opts.output_quantization_params.end());

    const QuantizationParams& param = it->second;

    Value* scale = gb->Const(runtime::MakeScalarArray(param.scale));
    Value* zero_point = gb->Const(runtime::MakeDtypeScalarArray(param.zero_point_dtype.chx(), param.zero_point));
    return {output, scale, zero_point, {}, {}};
}

std::vector<QuantizedInput> QuantizeInputs(
        const QuantizationOptions& opts, GraphBuilder* gb, Node* node, const std::vector<int64_t>& indices, int64_t weight_index) {
    CHECK(node->op_type() == Node::kConv || node->op_type() == Node::kMatMul);

    std::vector<QuantizedInput> result;

    for (int64_t input_index : indices) {
        Dtype qType = input_index == weight_index ? opts.weight_qdtype : opts.input_qdtype;
        Value* node_input = node->input(input_index);
        const Tensor* initializer = node_input->GetConstTensor();
        if (initializer) {
            // Treat input with initializer as weight
            QuantizedInput weight;
            if (node->op_type() == Node::kConv && input_index == weight_index) {
                weight = QuantizeWeightConvolution(opts, gb, initializer->chx(), qType);
            } else {
                weight = QuantizeWeight(opts, gb, initializer->chx(), qType);
            }

            if (!IsQuantized(weight)) {
                UpdateUnsupportedNodesUsingWeight(weight);
            }

            result.push_back(weight);
        } else {
            // Add QuantizeLiner
            Value* scale;
            Value* zero_point;
            if (opts.is_static) {
                auto it = opts.input_quantization_params.find(node_input->name());
                CHECK(it != opts.input_quantization_params.end());
                const QuantizationParams& param = it->second;
                scale = gb->Const(runtime::MakeScalarArray(param.scale));
                zero_point = gb->Const(runtime::MakeDtypeScalarArray(qType.chx(), param.zero_point));
            } else {
                // Graph for dynamic quantize parameter
                DataMode mode = ModeForDataType(qType);

                Value* rmin = gb->Op(Node::kReduceMin, {node_input});
                rmin->producer()->set_keep_dims(0);
                Value* rmax = gb->Op(Node::kReduceMax, {node_input});
                rmax->producer()->set_keep_dims(0);

                Value* fixed_qrange_scaled = gb->Const(runtime::MakeScalarArray(QRangeForQtype(qType)));

                if (mode == DataMode::Linear_Scaled) {
                    Value* abs_rmin = gb->Op(Node::kAbs, rmin);
                    Value* abs_rmax = gb->Op(Node::kAbs, rmax);
                    Value* abs_max = gb->Op(NOde::kMax, {abs_rmin, abs_rmax});
                    scale = gb->Op(Node::kDiv, {abs_max, fixed_qrange_scaled});

                    zero_point = gb->Const(runtime::MakeScalarArray(0.f));
                } else {
                    CHECK_EQ(DataMode::Linear_NonScaled, mode);

                    Value* scale_sub = gb->Op(Node::kSub, {rmax, rmin});
                    scale = gb->Op(Node::kDiv, {scale_sub, fixed_qrange_scaled});

                    Value* zp_sub = gb->Op(Node::kSub, {gb->Const(gb->Const(runtime::MakeScalarArray(0.f))), rmin});
                    Value* zp_div = gb->Op(Node::kDiv, {zp_sub, scale});
                    Value* zp_floor = gb->Op(Node::kFloor, {zp_div});
                    zero_point = gb->Op(Node::kCast, {zp_floor});
                }
            }

            Value* qlinear_out = gb->Op(Node::kQuantizeLinear, {node_input, scale, zero_point});
            result.push_back({qlinear_out, scale, zero_point});
        }
    }

    return result;
}

bool QuantizeConvolutionInteger(const QuantizationOptions& opts, Graph* graph, Node* conv) {
    CHECK_EQ(Node::kMatMul, conv->op_type());

    GraphBuilder gb(graph, "QuantizeConvWithInteger", conv->input(0));

    std::vector<QuantizedInput> quantized_inputs = QuantizeInputs(opts, &gb, conv, {0, 1}, 1);

    Value* conv_int_out =
            gb.Op(Node::kConvInteger,
                  {
                          quantized_inputs[0].input,
                          quantized_inputs[1].input,
                          quantized_inputs[0].zero_point,
                          quantized_inputs[1].zero_point,
                  });

    // Cast ConvInteger output to float
    Value* cast_out = gb.Op(Node::kCast, {conv_int_out});
    cast_out->producer()->set_to(Dtype(Dtype::kFloat32));

    // Scale back
    Value* scales_mul_op_out = gb.Op(Node::kMul, {quantized_inputs[0].scale, quantized_inputs[1].scale});
    gb.Op(Node::kMul, {cast_out, scales_mul_op_out}, conv->output(0));

    conv->Detach();

    return true;
}

bool QuantizeMatMulInteger(const QuantizationOptions& opts, Graph* graph, Node* matmul) {
    CHECK_EQ(Node::kMatMul, matmul->op_type());

    GraphBuilder gb(graph, "QuantizeMatMulWithInteger", matmul->input(0));

    std::vector<QuantizedInput> quantized_inputs = QuantizeInputs(opts, &gb, matmul, {0, 1}, 1);

    Value* matmul_int_out =
            gb.Op(Node::kMatMulInteger,
                  {
                          quantized_inputs[0].input,
                          quantized_inputs[1].input,
                          quantized_inputs[0].zero_point,
                          quantized_inputs[1].zero_point,
                  });

    // Cast MatMulInteger output to float
    Value* cast_out = gb.Op(Node::kCast, {matmul_int_out});
    cast_out->producer()->set_to(Dtype(Dtype::kFloat32));

    // Scale back
    Value* scales_mul_op_out = gb.Op(Node::kMul, {quantized_inputs[0].scale, quantized_inputs[1].scale});
    gb.Op(Node::kMul, {cast_out, scales_mul_op_out}, matmul->output(0));

    matmul->Detach();

    return true;
}

bool QuantizeConvolutionQLinear(const QuantizationOptions& opts, Graph* graph, Node* conv) {
    CHECK_EQ(Node::kConv, conv->op_type());

    GraphBuilder gb(graph, "QuantizeConvWithQLinear", conv->input(0));

    std::vector<QuantizedInput> quantized_inputs = QuantizeInputs(opts, &gb, conv, {0, 1}, 1);
    QuantizedOutput quantized_output = QuantizeOutput(opts, &gb, conv->output(0));

    Value* qlinear_conv_out =
            gb.Op(Node::kQLinearConv,
                  {
                          quantized_inputs[0].input,
                          quantized_inputs[0].scale,
                          quantized_inputs[0].zero_point,
                          quantized_inputs[1].input,
                          quantized_inputs[1].scale,
                          quantized_inputs[1].zero_point,
                          quantized_output.scale,
                          quantized_output.zero_point,
                  });
    // Copy convolution attributes
    qlinear_conv_out->producer()
            ->set_dilations(conv->dilations())
            ->set_group(conv->group())
            ->set_kernel_shape(conv->kernel_shape())
            ->set_strides(conv->strides())
            ->set_auto_pad(conv->auto_pad())
            ->set_pads(conv->pads());

    gb.Op(Node::kDequantizeLinear, {qlinear_conv_out, quantized_output.scale, quantized_output.zero_point}, conv->output(0));

    conv->Detach();

    return true;
}

bool QuantizeMatMulQLinear(const QuantizationOptions& opts, Graph* graph, Node* matmul) {
    CHECK_EQ(Node::kMatMul, matmul->op_type());

    GraphBuilder gb(graph, "QuantizeMatMulWithQLinear", matmul->input(0));

    std::vector<QuantizedInput> quantized_inputs = QuantizeInputs(opts, &gb, matmul, {0, 1}, 1);
    QuantizedOutput quantized_output = QuantizeOutput(opts, &gb, matmul->output(0));

    Value* qlinear_matmul_out =
            gb.Op(Node::kQLinearMatMul,
                  {
                          quantized_inputs[0].input,
                          quantized_inputs[0].scale,
                          quantized_inputs[0].zero_point,
                          quantized_inputs[1].input,
                          quantized_inputs[1].scale,
                          quantized_inputs[1].zero_point,
                          quantized_output.scale,
                          quantized_output.zero_point,
                  });
    gb.Op(Node::kDequantizeLinear, {qlinear_matmul_out, quantized_output.scale, quantized_output.zero_point}, matmul->output(0));

    matmul->Detach();

    return true;
}

bool QuantizeConvolution(const QuantizationOptions& opts, Graph* graph, Node* conv) {
    CHECK_EQ(Node::kConv, conv->op_type());

    if (opts.mode == QuantizationMode::IntegerOps) {
        return QuantizeConvolutionInteger(opts, graph, conv);
    }

    CHECK_EQ(QuantizationMode::QLinearOps, opts.mode);
    return QuantizeConvolutionQLinear(opts, graph, conv);
}

bool QuantizeMatMul(const QuantizationOptions& opts, Graph* graph, Node* matmul) {
    CHECK_EQ(Node::kMatMul, matmul->op_type());

    if (opts.mode == QuantizationMode::IntegerOps) {
        return QuantizeMatMulInteger(opts, graph, matmul);
    }

    CHECK_EQ(QuantizationMode::QLinearOps, opts.mode);
    return QuantizeMatMulQLinear(opts, graph, matmul);
}

bool QuantizeModel(const QuantizationOptions& opts, Graph* graph) {
    bool result = false;

    for (Node* node : graph->GetLiveNodes()) {
        switch (node->op_type()) {
            case Node::kConv:
                result = result || QuantizeConvolution(opts, graph, node);
                break;
            case Node::kMatMul:
                result = result || QuantizeMatMul(opts, graph, node);
                break;
            default:
                break;
        }
    }

    return result;
}

}  // namespace

bool Quantize(const QuantizationOptions& opts_, Graph* graph) {
    QuantizationOptions opts = opts_;
    CHECK_EQ(8, opts.nbits);
    CHECK_EQ(QuantizationMethod::OnnxRuntime, opts.method);
    if (opts.input_qdtype == Dtype::kUnknown) {
        opts.input_qdtype = Dtype(Dtype::kInt8);
    }
    if (opts.weight_qdtype == Dtype::kUnknown) {
        opts.weight_qdtype = opts.asymmertic_input_types ? Dtype::kInt8 : Dtype::kUInt8;
    }

    return QuantizeModel(opts, graph);
}

}  // namespace chainer_compiler
