#pragma once

#include <string>
#include <unordered_map>

#include <compiler/dtype.h>

namespace chainer_compiler {

class Graph;

enum class QuantizationMethod {
    OnnxRuntime,
    // SNPE,
    // TensorRT,
    // Outlier Channel Splitting,
    // Clipping
};

enum class QuantizationMode {
    IntegerOps = 0,
    QLinearOps = 1,
};

struct QuantizationParams {
    Dtype zero_point_dtype;
    float zero_point, scale;
};

struct QuantizationOptions {
    QuantizationMethod method = QuantizationMethod::OnnxRuntime;

    int nbits = 8;
    // TODO(take-cheeze): Make this true
    bool per_channel = false;
    QuantizationMode mode = QuantizationMode::IntegerOps;
    bool is_static = false;
    bool asymmertic_input_types = false;

    std::unordered_map<std::string, QuantizationParams> input_quantization_params;
    std::unordered_map<std::string, QuantizationParams> output_quantization_params;
};

bool Quantize(const QuantizationOptions& opts, Graph* graph);

std::ostream& operator<<(std::ostream& os, QuantizationMode mode);
std::ostream& operator<<(std::ostream& os, QuantizationMethod meth);

}  // namespace chainer_compiler
