#pragma once

#include <iosfwd>

#include <onnx/onnx.pb.h>

namespace oniku {

class Dtype {
public:
    // These values must be synchronized with xChainer's.
    enum DataType {
        kUnspecified = 0,
        kBool = 1,
        kInt8,
        kInt16,
        kInt32,
        kInt64,
        kUInt8,
        kFloat32,
        kFloat64,
    };

    Dtype() = default;
    explicit Dtype(const onnx::TensorProto::DataType& xtype);
    // Note this is an implicit constructor.
    Dtype(DataType type);

    operator DataType() const {
        return type_;
    }

    onnx::TensorProto::DataType ToONNX() const;
    std::string ToString() const;

    int SizeOf() const;

private:
    DataType type_ = kUnspecified;
};

std::ostream& operator<<(std::ostream& os, const Dtype& dtype);

}  // namespace oniku
