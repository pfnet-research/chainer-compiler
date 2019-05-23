#include "tools/util.h"

#include <algorithm>

#include <chainerx/array.h>
#include <chainerx/dtype.h>
#include <chainerx/error.h>
#include <chainerx/indexable_array.h>
#include <chainerx/indexer.h>
#include <chainerx/native/data_type.h>
#include <chainerx/numeric.h>

#include <compiler/graph.h>
#include <compiler/model.h>
#include <runtime/chainerx_util.h>
#include <runtime/chxvm_var.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Dtype ChainerXTypeFromONNX(int xtype) {
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
        case onnx::TensorProto::FLOAT16:
            return chainerx::Dtype::kFloat16;
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
            chainerx::Dtype dtype = ChainerXTypeFromONNX(initializer->dtype().ToONNX());
            chainerx::Shape shape(initializer->dims());
            const void* data = initializer->GetRawData();
            chainerx::Array tensor;
            // If the input is used only by Reshape as a shape, place
            // it on host memory.
            // TODO(hamaji): Introduce more sophisticated approach to
            // decide the device to be used.
            if (std::find_if(input->users().begin(), input->users().end(), [input](const Node* node) {
                    return node->op_type() != Node::kReshape || node->input(1) != input;
                }) == input->users().end()) {
                tensor = MakeHostArray(dtype, shape, data);
            } else {
                tensor = MakeArray(dtype, shape, data);
            }
            CHECK(params.emplace(initializer->name(), std::shared_ptr<ChxVMVar>(new ChxVMVar(tensor))).second)
                    << "Duplicate input tensor: " << initializer->name();
        }
    }
    return params;
}

int MismatchInAllClose(const chainerx::Array& a, const chainerx::Array& b, double rtol, double atol, bool equal_nan) {
    // Most part of this code is copied from chainerx
    if (a.shape() != b.shape()) {
        throw chainerx::DimensionError{"Cannot compare Arrays of different shapes: ", a.shape(), ", ", b.shape()};
    }
    if (a.dtype() != b.dtype()) {
        throw chainerx::DtypeError{"Cannot compare Arrays of different Dtypes: ", a.dtype(), ", ", b.dtype()};
    }

    chainerx::Array a_native = a.ToNative();
    chainerx::Array b_native = b.ToNative();

    return VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        chainerx::IndexableArray<const T> a_iarray{a_native};
        chainerx::IndexableArray<const T> b_iarray{b_native};
        chainerx::Indexer<> indexer{a_native.shape()};

        int64_t error_count = 0;
        for (auto it = indexer.It(0); it; ++it) {
            T ai = chainerx::native::StorageToDataType<const T>(a_iarray[it]);
            T bi = chainerx::native::StorageToDataType<const T>(b_iarray[it]);
            if (equal_nan && chainerx::IsNan(ai) && chainerx::IsNan(bi)) {
                // nop
            } else if (
                    chainerx::IsNan(ai) || chainerx::IsNan(bi) ||
                    std::abs(static_cast<double>(ai) - static_cast<double>(bi)) > atol + rtol * std::abs(static_cast<double>(bi))) {
                error_count++;
            }
        }
        return error_count;
    });
}

}  // namespace runtime
}  // namespace chainer_compiler
