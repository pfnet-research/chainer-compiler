#pragma once

#include <string>
#include <vector>

#include <onnx/onnx.pb.h>

#include <compiler/dtype.h>

namespace oniku {

class Type {
public:
    explicit Type(const onnx::TypeProto& xtype);
    Type(Dtype dtype, const std::vector<int>& dims);

    void ToONNX(onnx::TypeProto* xtype) const;

private:
    Dtype dtype_;
    std::vector<int> dims_;
    std::vector<std::string> dim_params_;
    std::vector<std::string> denotations_;
};

}  // namespace oniku
