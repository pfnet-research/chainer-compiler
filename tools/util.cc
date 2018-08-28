#include "util.h"

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/tensor.h>
#include <compiler/value.h>
#include <runtime/xchainer.h>

namespace oniku {
namespace runtime {

void MakeHumanReadableValue(onnx::TensorProto* tensor) {
    if (tensor->raw_data().empty())
        return;
    Tensor t(*tensor);
    tensor->Clear();
    t.ToONNX(tensor);
}

void StripLargeValue(onnx::TensorProto* tensor, int num_elements) {
#define CLEAR_IF_LARGE(tensor, x, n) do {                               \
        if (tensor->x().size() >= n) {                                  \
            auto msg = StrCat("* ", tensor->x().size(), " elements cleared *"); \
            tensor->add_string_data(msg);                               \
            tensor->clear_##x();                                        \
        }                                                               \
    } while(0)
    CLEAR_IF_LARGE(tensor, float_data, num_elements);
    CLEAR_IF_LARGE(tensor, int32_data, num_elements);
    CLEAR_IF_LARGE(tensor, int64_data, num_elements);
    CLEAR_IF_LARGE(tensor, raw_data, num_elements * 4);
    CLEAR_IF_LARGE(tensor, double_data, num_elements);
    CLEAR_IF_LARGE(tensor, uint64_data, num_elements);
}

xchainer::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype) {
    switch (xtype) {
        case onnx::TensorProto::BOOL:
            return xchainer::Dtype::kBool;
        case onnx::TensorProto::INT8:
            return xchainer::Dtype::kInt8;
        case onnx::TensorProto::INT16:
            return xchainer::Dtype::kInt16;
        case onnx::TensorProto::INT32:
            return xchainer::Dtype::kInt32;
        case onnx::TensorProto::INT64:
            return xchainer::Dtype::kInt64;
        case onnx::TensorProto::UINT8:
            return xchainer::Dtype::kUInt8;
        case onnx::TensorProto::FLOAT:
            return xchainer::Dtype::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return xchainer::Dtype::kFloat64;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

InOuts LoadParams(const Model& model) {
    InOuts params;
    for (const Value* input : model.graph().input_values()) {
        if (const Tensor* initializer = input->initializer()) {
            xchainer::Dtype dtype = XChainerTypeFromONNX(initializer->dtype().ToONNX());
            xchainer::Shape shape(initializer->dims());
            const void* data = initializer->GetRawData();
            xchainer::Array tensor(MakeArray(dtype, shape, data));
            CHECK(params.emplace(initializer->name(), tensor).second) << "Duplicate input tensor: " << initializer->name();
        }
    }
    return params;
}

}  // namespace runtime
}  // namespace oniku
