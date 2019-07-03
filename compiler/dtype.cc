#include "compiler/dtype.h"

#include <common/log.h>
#include <common/strutil.h>

namespace chainer_compiler {

namespace {

Dtype::DataType FromONNX(int xtype) {
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
        case onnx::TensorProto::FLOAT16:
            return Dtype::DataType::kFloat16;
        case onnx::TensorProto::FLOAT:
            return Dtype::DataType::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return Dtype::DataType::kFloat64;
        case onnx::TensorProto::UNDEFINED:
            return Dtype::DataType::kUnknown;
        case onnx::TensorProto::STRING:
            return Dtype::DataType::kString;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

Dtype::DataType FromChainerX(chainerx::Dtype type) {
    switch (type) {
        case chainerx::Dtype::kBool:
            return Dtype::DataType::kBool;
        case chainerx::Dtype::kInt8:
            return Dtype::DataType::kInt8;
        case chainerx::Dtype::kInt16:
            return Dtype::DataType::kInt16;
        case chainerx::Dtype::kInt32:
            return Dtype::DataType::kInt32;
        case chainerx::Dtype::kInt64:
            return Dtype::DataType::kInt64;
        case chainerx::Dtype::kUInt8:
            return Dtype::DataType::kUInt8;
        case chainerx::Dtype::kFloat16:
            return Dtype::DataType::kFloat16;
        case chainerx::Dtype::kFloat32:
            return Dtype::DataType::kFloat32;
        case chainerx::Dtype::kFloat64:
            return Dtype::DataType::kFloat64;
        default:
            CHECK(false) << "Unknown ChainerX data type: " << type;
    }
}

}  // namespace

Dtype::Dtype(int xtype) : type_(FromONNX(xtype)) {
}

Dtype::Dtype(DataType type) : type_(type) {
}

Dtype::Dtype(chainerx::Dtype type) : type_(FromChainerX(type)) {
}

chainerx::Dtype Dtype::chx() const {
    switch (type_) {
        case kBool:
            return chainerx::Dtype::kBool;
        case kInt8:
            return chainerx::Dtype::kInt8;
        case kInt16:
            return chainerx::Dtype::kInt16;
        case kInt32:
            return chainerx::Dtype::kInt32;
        case kInt64:
            return chainerx::Dtype::kInt64;
        case kUInt8:
            return chainerx::Dtype::kUInt8;
        case kFloat16:
            return chainerx::Dtype::kFloat16;
        case kFloat32:
            return chainerx::Dtype::kFloat32;
        case kFloat64:
            return chainerx::Dtype::kFloat64;
        default:
            CHECK(false) << "Unknown ChainerX data type: " << type_;
    }
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
        case kFloat16:
            return "FLOAT16";
        case kFloat32:
            return "FLOAT32";
        case kFloat64:
            return "FLOAT64";
        case kString:
            return "STRING";
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
        case kFloat16:
            return onnx::TensorProto::FLOAT16;
        case kFloat32:
            return onnx::TensorProto::FLOAT;
        case kFloat64:
            return onnx::TensorProto::DOUBLE;
        case kString:
            return onnx::TensorProto::STRING;
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
        case kFloat16:
            return 2;
        case kFloat32:
            return 4;
        case kFloat64:
            return 8;
        case kString:
            // TODO(take-cheeze): Make size accurate
            return 0;
        default:
            CHECK(false) << "Unknown data type: " << ToString();
    }
}

std::ostream& operator<<(std::ostream& os, const Dtype& dtype) {
    os << dtype.ToString();
    return os;
}

}  // namespace chainer_compiler
