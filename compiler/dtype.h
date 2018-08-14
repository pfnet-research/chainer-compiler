#pragma once

#include <onnx/onnx.pb.h>

namespace oniku {

class Dtype {
public:
    enum DataType {
        kBool = 1,
        kInt8,
        kInt16,
        kInt32,
        kInt64,
        kUInt8,
        kFloat32,
        kFloat64,
    };

    explicit Dtype(const onnx::TensorProto::DataType& xtype);
    explicit Dtype(DataType type);

    operator DataType() const {
        return type_;
    }

    onnx::TensorProto::DataType ToONNX() const;
    std::string ToString() const;

    int SizeOf() const;

private:
    DataType type_;
};

}  // namespace oniku
