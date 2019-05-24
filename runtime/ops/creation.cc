#include <chainerx/native/native_backend.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/manipulation.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

chainerx::Array IntScalarConstantOp::RunImpl(ChxVMState* st) {
    chainerx::Device& device = host ? chainerx::GetNativeBackend().GetDevice(0) : chainerx::GetDefaultDevice();
    return chainerx::Full({}, value, static_cast<chainerx::Dtype>(dtype), device);
}

chainerx::Array FloatScalarConstantOp::RunImpl(ChxVMState* st) {
    chainerx::Device& device = host ? chainerx::GetNativeBackend().GetDevice(0) : chainerx::GetDefaultDevice();
    return chainerx::Full({}, value, static_cast<chainerx::Dtype>(dtype), device);
}

class IntConstantOp::IntConstantImpl {
public:
    chainerx::Array cache;
};

void IntConstantOp::InitImpl() {
    impl_ = new IntConstantImpl();
    auto make = host ? MakeHostArray : MakeArray;
    chainerx::Array a = make(chainerx::Dtype::kInt64, chainerx::Shape(shape), value.data());
    impl_->cache = a.AsType(static_cast<chainerx::Dtype>(dtype));
}

IntConstantOp::~IntConstantOp() {
    delete impl_;
}

class FloatConstantOp::FloatConstantImpl {
public:
    chainerx::Array cache;
};

void FloatConstantOp::InitImpl() {
    impl_ = new FloatConstantImpl();
    auto make = host ? MakeHostArray : MakeArray;
    chainerx::Array a = make(chainerx::Dtype::kFloat64, chainerx::Shape(shape), value.data());
    impl_->cache = a.AsType(static_cast<chainerx::Dtype>(dtype));
}

FloatConstantOp::~FloatConstantOp() {
    delete impl_;
}

chainerx::Array IntConstantOp::RunImpl(ChxVMState* st) {
    return impl_->cache;
}

chainerx::Array FloatConstantOp::RunImpl(ChxVMState* st) {
    return impl_->cache;
}

chainerx::Array OneHotOp::RunImpl(
        ChxVMState* st, const chainerx::Array& indices, const chainerx::Array& depth, const chainerx::Array& values) {
    int rank = indices.ndim();
    chainerx::Array depth_range = chainerx::Arange(chainerx::AsScalar(depth), indices.device());
    int axis = this->axis;
    if (axis < 0) axis += rank + 1;

    chainerx::Shape targets_shape;
    chainerx::Shape values_shape;
    for (int i = 0; i < axis; ++i) {
        targets_shape.push_back(1);
        values_shape.push_back(indices.shape()[i]);
    }
    targets_shape.push_back(depth_range.shape()[0]);
    values_shape.push_back(1);
    for (int i = axis; i < rank; ++i) {
        targets_shape.push_back(1);
        values_shape.push_back(indices.shape()[i]);
    }
    chainerx::Array targets = chainerx::Reshape(depth_range, targets_shape);

    chainerx::Array mask = (targets.AsType(indices.dtype()) == chainerx::Reshape(indices, values_shape));
    mask = mask.AsType(values.dtype());

    chainerx::Scalar off_value = chainerx::AsScalar(values.At({0}));
    chainerx::Scalar on_value = chainerx::AsScalar(values.At({1}));
    return mask * (on_value + (-off_value)) + off_value;
}

chainerx::Array ConstantFillOp::RunImpl(ChxVMState* st, const nonstd::optional<chainerx::Array>& input) {
    CHECK(extra_shape.empty()) << "extra_shape not implemented yet";
    chainerx::Dtype dtype = this->dtype ? static_cast<chainerx::Dtype>(this->dtype) : chainerx::Dtype::kFloat32;
    chainerx::Shape shape;
    if (input.has_value()) {
        shape = ArrayToShape(*input);
    } else {
        shape = chainerx::Shape(this->shape);
    }
    return chainerx::Full(shape, value, dtype);
}

chainerx::Array EyeLikeOp::RunImpl(ChxVMState* st, const chainerx::Array& input) {
    chainerx::Dtype dtype = this->dtype ? static_cast<chainerx::Dtype>(this->dtype) : input.dtype();
    return chainerx::Eye(input.shape()[0], input.shape()[1], this->k, dtype, input.device());
}

}  // namespace runtime
}  // namespace chainer_compiler
