#include "type.h"

#include <common/log.h>

namespace oniku {

Type::Type(Kind kind) : kind_(kind) {
    has_known_shape_ = false;
}

Type::Type(const onnx::TypeProto& xtype) {
    if (xtype.has_sequence_type()) {
        kind_ = Kind::kSequence;
        has_known_shape_ = false;
        sequence_.reset(new Type(xtype.sequence_type().elem_type()));
        return;
    }

    if (xtype.has_map_type()) {
        CHECK(false) << "ONNX map not implemented yet";
    }

    if (xtype.has_opaque_type()) {
        kind_ = Kind::kOpaque;
        has_known_shape_ = false;
        return;
    }

    CHECK(xtype.has_tensor_type()) << xtype.DebugString();
    dtype_ = Dtype(xtype.tensor_type().elem_type());
    has_known_shape_ = xtype.tensor_type().has_shape();
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

Type::Type(Dtype dtype, const std::vector<int64_t>& dims) : dtype_(dtype), dims_(dims) {
}

Type::Type(const Type& type)
    : kind_(type.kind_),
      dtype_(type.dtype_),
      dims_(type.dims_),
      dim_params_(type.dim_params_),
      denotations_(type.denotations_),
      sequence_(type.sequence_.get() ? new Type(*type.sequence_) : nullptr),
      has_known_shape_(type.has_known_shape_) {
}

void Type::ToONNX(onnx::TypeProto* xtype) const {
    if (kind_ == Kind::kSequence) {
        sequence_->ToONNX(xtype->mutable_sequence_type()->mutable_elem_type());
        return;
    }
    CHECK(sequence_.get() == nullptr);

    if (kind_ == Kind::kMap) {
        CHECK(false) << "ONNX map not implemented yet";
    }

    if (kind_ == Kind::kOpaque) {
        xtype->mutable_opaque_type();
        return;
    }

    xtype->mutable_tensor_type()->set_elem_type(dtype_.ToONNX());
    if (!has_known_shape_) return;
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

std::string Type::DebugString() const {
    onnx::TypeProto xtype;
    ToONNX(&xtype);
    return xtype.DebugString();
}

int64_t Type::NumElements() const {
    CHECK_EQ(kind_, Kind::kTensor);
    if (!has_known_shape_) return -1;
    int64_t num = 1;
    for (int d : dims_) {
        if (d < 0) return -1;
        num *= d;
    }
    return num;
}

int64_t Type::GetNBytes() const {
    if (dtype_ == Dtype::kUnknown) return -1;
    int64_t num_elements = NumElements();
    if (num_elements < 0) return -1;
    return num_elements * dtype_.SizeOf();
}

std::ostream& operator<<(std::ostream& os, const Type::Kind& kind) {
    static const char* kNames[] = {"Tensor", "Sequence", "Map", "Opaque"};
    int k = static_cast<int>(kind);
    if (k >= 0 && k < sizeof(kNames) / sizeof(kNames[0])) {
        os << kNames[k];
    } else {
        os << "???(" << k << ")";
    }
    return os;
}

}  // namespace oniku
