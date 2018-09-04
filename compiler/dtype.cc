#include "dtype.h"

#include <common/log.h>
#include <common/strutil.h>

namespace oniku {

namespace {

Dtype::DataType FromONNX(onnx::TensorProto::DataType xtype) {
    switch (xtype) {
        case onnx::TensorProto::BOOL:
            return Dtype::DataType::kBool;
        case onnx::TensorProto::INT8:
            return Dtype::DataType::kInt8;
        case onnx::TensorProto::INT16:
            return Dtype::DataType::kInt16;
        case onnx::TensorProto::INT32:
            return Dtype::DataType::kInt32;
        case onnx::TensorProto::INT64:
            return Dtype::DataType::kInt64;
        case onnx::TensorProto::UINT8:
            return Dtype::DataType::kUInt8;
        case onnx::TensorProto::FLOAT:
            return Dtype::DataType::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return Dtype::DataType::kFloat64;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

}  // namespace

Dtype::Dtype(const onnx::TensorProto::DataType& xtype) : type_(FromONNX(xtype)) {
}

Dtype::Dtype(DataType type) : type_(type) {
}

std::string Dtype::ToString() const {
    switch (type_) {
        case kBool:
            return "BOOL";
        case kInt8:
            return "INT8";
        case kInt16:
            return "INT16";
        case kInt32:
            return "INT32";
        case kInt64:
            return "INT64";
        case kUInt8:
            return "UINT8";
        case kFloat32:
            return "FLOAT32";
        case kFloat64:
            return "FLOAT64";
        case kUnknown:
            return "UNKNOWN";
        default:
            return StrCat("?(", static_cast<int>(type_), ")");
    }
}

onnx::TensorProto::DataType Dtype::ToONNX() const {
    switch (type_) {
        case kBool:
            return onnx::TensorProto::BOOL;
        case kInt8:
            return onnx::TensorProto::INT8;
        case kInt16:
            return onnx::TensorProto::INT16;
        case kInt32:
            return onnx::TensorProto::INT32;
        case kInt64:
            return onnx::TensorProto::INT64;
        case kUInt8:
            return onnx::TensorProto::UINT8;
        case kFloat32:
            return onnx::TensorProto::FLOAT;
        case kFloat64:
            return onnx::TensorProto::DOUBLE;
        case kUnknown:
            return onnx::TensorProto::UNDEFINED;
        default:
            CHECK(false) << "Unknown data type: " << ToString();
    }
}

int Dtype::SizeOf() const {
    switch (type_) {
        case kBool:
            return 1;
        case kInt8:
            return 1;
        case kInt16:
            return 2;
        case kInt32:
            return 4;
        case kInt64:
            return 8;
        case kUInt8:
            return 1;
        case kFloat32:
            return 4;
        case kFloat64:
            return 8;
        default:
            CHECK(false) << "Unknown data type: " << ToString();
    }
}

std::ostream& operator<<(std::ostream& os, const Dtype& dtype) {
    os << dtype.ToString();
    return os;
}

}  // namespace oniku
