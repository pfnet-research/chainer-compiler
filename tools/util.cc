#include "tools/util.h"

#include <algorithm>

#include <compiler/graph.h>
#include <compiler/model.h>
#include <runtime/xcvm_var.h>

namespace oniku {
namespace runtime {

chainerx::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype) {
    switch (xtype) {
        case onnx::TensorProto::BOOL:
            return chainerx::Dtype::kBool;
        case onnx::TensorProto::INT8:
            return chainerx::Dtype::kInt8;
        case onnx::TensorProto::INT16:
            return chainerx::Dtype::kInt16;
        case onnx::TensorProto::INT32:
            return chainerx::Dtype::kInt32;
        case onnx::TensorProto::INT64:
            return chainerx::Dtype::kInt64;
        case onnx::TensorProto::UINT8:
            return chainerx::Dtype::kUInt8;
        case onnx::TensorProto::FLOAT:
            return chainerx::Dtype::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return chainerx::Dtype::kFloat64;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

InOuts LoadParams(const Graph& graph) {
    InOuts params;
    for (const Value* input : graph.input_values()) {
        if (input->users().empty()) continue;
        if (const Tensor* initializer = input->initializer()) {
            chainerx::Dtype dtype = XChainerTypeFromONNX(initializer->dtype().ToONNX());
            chainerx::Shape shape(initializer->dims());
            const void* data = initializer->GetRawData();
            chainerx::Array tensor;
            // If the input is used only by Reshape as a shape, place
            // it on host memory.
            // TODO(hamaji): Introduce more sophisticated approach to
            // decide the device to be used.
            if (std::find_if(input->users().begin(), input->users().end(), [input](const Node* node) {
                    return node->op_type() != Node::kReshape || node->inputs()[1] != input;
                }) == input->users().end()) {
                tensor = MakeHostArray(dtype, shape, data);
            } else {
                tensor = MakeArray(dtype, shape, data);
            }
            CHECK(params.emplace(initializer->name(), std::shared_ptr<XCVMVar>(new XCVMVar(tensor))).second)
                    << "Duplicate input tensor: " << initializer->name();
        }
    }
    return params;
}

}  // namespace runtime
}  // namespace oniku
