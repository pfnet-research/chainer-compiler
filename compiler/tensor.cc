#include "tensor.h"

#include <cstdlib>
#include <sstream>

#include <common/log.h>

namespace oniku {

namespace {

Tensor::Dtype FromONNX(onnx::TensorProto::DataType xtype) {
    switch (xtype) {
        case onnx::TensorProto::BOOL:
            return Tensor::Dtype::kBool;
        case onnx::TensorProto::INT8:
            return Tensor::Dtype::kInt8;
        case onnx::TensorProto::INT16:
            return Tensor::Dtype::kInt16;
        case onnx::TensorProto::INT32:
            return Tensor::Dtype::kInt32;
        case onnx::TensorProto::INT64:
            return Tensor::Dtype::kInt64;
        case onnx::TensorProto::UINT8:
            return Tensor::Dtype::kUInt8;
        case onnx::TensorProto::FLOAT:
            return Tensor::Dtype::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return Tensor::Dtype::kFloat64;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

std::string ToString(Tensor::Dtype type) {
    switch (type) {
        case Tensor::Dtype::kBool:
            return "BOOL";
        case Tensor::Dtype::kInt8:
            return "INT8";
        case Tensor::Dtype::kInt16:
            return "INT16";
        case Tensor::Dtype::kInt32:
            return "INT32";
        case Tensor::Dtype::kInt64:
            return "INT64";
        case Tensor::Dtype::kUInt8:
            return "UINT8";
        case Tensor::Dtype::kFloat32:
            return "FLOAT32";
        case Tensor::Dtype::kFloat64:
            return "FLOAT64";
        default: {
            std::ostringstream oss;
            oss << "???(" << static_cast<int>(type) << ")";
            return oss.str();
        }
    }
}

int SizeOf(Tensor::Dtype type) {
    switch (type) {
        case Tensor::Dtype::kBool:
            return 1;
        case Tensor::Dtype::kInt8:
            return 1;
        case Tensor::Dtype::kInt16:
            return 2;
        case Tensor::Dtype::kInt32:
            return 4;
        case Tensor::Dtype::kInt64:
            return 8;
        case Tensor::Dtype::kUInt8:
            return 1;
        case Tensor::Dtype::kFloat32:
            return 4;
        case Tensor::Dtype::kFloat64:
            return 8;
        default:
            CHECK(false) << "Unknown data type: " << ToString(type);
    }
}

template <typename From, typename To>
Tensor::UniqueData LoadDataFromRepeated(const ::google::protobuf::RepeatedField<From>& a) {
    Tensor::UniqueData p(std::malloc(sizeof(To) * a.size()), &std::free);
    for (int i = 0; i < a.size(); ++i) {
        static_cast<To*>(p.get())[i] = a.Get(i);
    }
    return p;
}

}  // namespace

Tensor::Tensor(const onnx::TensorProto& xtensor)
    : dims_(xtensor.dims().begin(), xtensor.dims().end()),
      dtype_(FromONNX(xtensor.data_type())),
      data_(nullptr, &std::free),
      name_(xtensor.name()),
      doc_string_(xtensor.doc_string()) {
    CHECK(!xtensor.has_segment()) << "Segmented TensorProto not supported";
    // TODO(hamaji): Support this.
    CHECK(!xtensor.has_raw_data()) << "raw_data is not supported yet";

    switch (dtype_) {
        case Tensor::Dtype::kBool:
            data_.reset(LoadDataFromRepeated<int32_t, bool>(xtensor.int32_data()).release());
            break;
        case Tensor::Dtype::kInt8:
            data_.reset(LoadDataFromRepeated<int32_t, int8_t>(xtensor.int32_data()).release());
            break;
        case Tensor::Dtype::kInt16:
            data_.reset(LoadDataFromRepeated<int32_t, int16_t>(xtensor.int32_data()).release());
            break;
        case Tensor::Dtype::kInt32:
            data_.reset(LoadDataFromRepeated<int32_t, int32_t>(xtensor.int32_data()).release());
            break;
        case Tensor::Dtype::kInt64:
            data_.reset(LoadDataFromRepeated<int64_t, int64_t>(xtensor.int64_data()).release());
            break;
        case Tensor::Dtype::kUInt8:
            data_.reset(LoadDataFromRepeated<int32_t, uint8_t>(xtensor.int32_data()).release());
            break;
        case Tensor::Dtype::kFloat32:
            data_.reset(LoadDataFromRepeated<float, float>(xtensor.float_data()).release());
            break;
        case Tensor::Dtype::kFloat64:
            data_.reset(LoadDataFromRepeated<double, double>(xtensor.double_data()).release());
            break;
        default:
            CHECK(false) << "Unknown data type: " << ToString(dtype_);
    }
}

Tensor::~Tensor() {}

void Tensor::ToONNX(onnx::TensorProto* xtensor) { CHECK(false) << "TODO"; }

int64_t Tensor::NumElements() const {
    int num = 1;
    for (int d : dims_) {
        if (d < 0) return -1;
        num *= d;
    }
    return num;
}

}  // namespace oniku
