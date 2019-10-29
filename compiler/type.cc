#include "compiler/type.h"

#include <sstream>

#include <common/log.h>
#include <common/strutil.h>

namespace chainer_compiler {

Type::Type(Kind kind) : kind_(kind) {
    has_known_shape_ = false;
}

Type::Type(const onnx::TypeProto& xtype) : denotation_(xtype.denotation()) {
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
            dim_denotations_.resize(dims_.size());
            dim_denotations_.push_back(dim.denotation());
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

Type::Type() : Type(Dtype::kUnknown) {
    has_known_shape_ = false;
}

Type::Type(Dtype dtype) : dtype_(dtype) {
    has_known_shape_ = false;
}

Type::Type(Dtype dtype, const std::vector<int64_t>& dims) : dtype_(dtype), dims_(dims) {
}

Type::Type(const Type& type)
    : kind_(type.kind_),
      dtype_(type.dtype_),
      dims_(type.dims_),
      dim_params_(type.dim_params_),
      dim_denotations_(type.dim_denotations_),
      sequence_(type.sequence_.get() ? new Type(*type.sequence_) : nullptr),
      denotation_(type.denotation()),
      has_known_shape_(type.has_known_shape_) {
}

void Type::ToONNX(onnx::TypeProto* xtype) const {
    if (!denotation_.empty()) {
        xtype->set_denotation(denotation_);
    }

    if (kind_ == Kind::kSequence) {
        if (sequence_.get()) {
            sequence_->ToONNX(xtype->mutable_sequence_type()->mutable_elem_type());
        } else {
            Type().ToONNX(xtype->mutable_sequence_type()->mutable_elem_type());
        }
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
        if (i < dim_denotations_.size() && !dim_denotations_[i].empty()) {
            dim->set_denotation(dim_denotations_[i]);
        }
    }
}

std::string Type::DebugString() const {
    onnx::TypeProto xtype;
    ToONNX(&xtype);
    return xtype.DebugString();
}

std::string Type::ToString() const {
    std::ostringstream oss;
    switch (kind_) {
        case Kind::kTensor: {
            std::string shape_str;
            if (has_known_shape_) {
                shape_str = JoinString(MapToString(dims_, [](int64_t d) { return d < 0 ? "?" : StrCat(d); }), ",");
            } else {
                shape_str = "UNKNOWN";
            }
            oss << "Tensor";
            oss << "(dtype=" << dtype_;
            oss << " shape=" << shape_str;
            oss << ")";
            break;
        }

        case Kind::kSequence:
            oss << "Sequence(";
            if (sequence_) {
                oss << sequence_->ToString();
            }
            oss << ")";
            break;

        case Kind::kMap:
            CHECK(false) << "Map not implemented";
            break;

        case Kind::kOpaque:
            oss << "Opaque()";
            break;
    }
    return oss.str();
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
    if (kind_ != Kind::kTensor) return -1;
    int64_t num_elements = NumElements();
    if (num_elements < 0) return -1;
    return num_elements * dtype_.SizeOf();
}

bool Type::HasKnownShape() const {
    if (!has_known_shape_) return false;
    CHECK_EQ(kind_, Kind::kTensor);
    for (int d : dims_) {
        if (d < 0) return false;
    }
    return true;
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

}  // namespace chainer_compiler
