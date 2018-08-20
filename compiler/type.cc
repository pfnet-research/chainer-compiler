#include "type.h"

#include <common/log.h>

namespace oniku {

Type::Type(const onnx::TypeProto& xtype) {
    CHECK(xtype.has_tensor_type()) << xtype.DebugString();
    dtype_ = Dtype(xtype.tensor_type().elem_type());
    for (const auto& dim : xtype.tensor_type().shape().dim()) {
        if (dim.has_denotation()) {
            denotations_.resize(dims_.size());
            denotations_.push_back(dim.denotation());
        }
        if (dim.has_dim_value()) {
            dims_.push_back(dim.dim_value());
        } else {
            dim_params_.resize(dims_.size());
            dim_params_.push_back(dim.dim_param());
            dims_.push_back(-1);
        }
    }
}

Type::Type(Dtype dtype, const std::vector<int>& dims) : dtype_(dtype), dims_(dims.begin(), dims.end()) {
}

void Type::ToONNX(onnx::TypeProto* xtype) const {
    xtype->mutable_tensor_type()->set_elem_type(dtype_.ToONNX());
    onnx::TensorShapeProto* xshape = xtype->mutable_tensor_type()->mutable_shape();
    for (size_t i = 0; i < dims_.size(); ++i) {
        auto* dim = xshape->add_dim();
        if (dims_[i] >= 0) {
            dim->set_dim_value(dims_[i]);
        } else if (i < dim_params_.size() && !dim_params_[i].empty()) {
            dim->set_dim_param(dim_params_[i]);
        }
        if (i < denotations_.size() && !denotations_[i].empty()) {
            dim->set_denotation(denotations_[i]);
        }
    }
}

int64_t Type::NumElements() const {
    int64_t num = 1;
    for (int d : dims_) {
        if (d < 0) return -1;
        num *= d;
    }
    return num;
}

}  // namespace oniku
