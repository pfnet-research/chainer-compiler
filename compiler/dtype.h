#pragma once

#include <iosfwd>

#include <chainerx/dtype.h>

#include <compiler/onnx.h>

namespace chainer_compiler {

class Dtype {
public:
    // These values must be synchronized with ChainerX's.
    enum DataType {
        kUnknown = 0,
        kBool = 1,
        kInt8,
        kInt16,
        kInt32,
        kInt64,
        kUInt8,
        kFloat16,
        kFloat32,
        kFloat64,
    };

    Dtype() = default;
    // Accepts `TensorProto::DataType` type.
    explicit Dtype(int xtype);
    // Note this is an implicit constructor.
    Dtype(DataType type);
    explicit Dtype(chainerx::Dtype type);

    operator DataType() const {
        return type_;
    }

    chainerx::Dtype chx() const;

    onnx::TensorProto::DataType ToONNX() const;
    std::string ToString() const;

    int SizeOf() const;

    bool IsFloat() const {
        return type_ == kFloat16 || type_ == kFloat32 || type_ == kFloat64;
    }

private:
    DataType type_ = kUnknown;
};

std::ostream& operator<<(std::ostream& os, const Dtype& dtype);

}  // namespace chainer_compiler
