#if CHAINER_COMPILER_ENABLE_NGRAPH

#include <sstream>

#include <chainerx/routines/creation.h>

#include <ngraph/frontend/onnx_import/onnx.hpp>
#include <ngraph/ngraph.hpp>

#include <common/log.h>

#else

#include <common/log.h>

#endif

#include <runtime/gen_xcvm_ops.h>

namespace chainer_compiler {
namespace runtime {

#if CHAINER_COMPILER_ENABLE_NGRAPH

namespace {

chainerx::Dtype GetDtype(ngraph::element::Type type) {
    switch (type.get_type_enum()) {
    case ngraph::element::Type_t::boolean:
        return chainerx::Dtype::kBool;
    case ngraph::element::Type_t::f32:
        return chainerx::Dtype::kFloat32;
    case ngraph::element::Type_t::f64:
        return chainerx::Dtype::kFloat64;
    case ngraph::element::Type_t::i8:
        return chainerx::Dtype::kInt8;
    case ngraph::element::Type_t::i16:
        return chainerx::Dtype::kInt16;
    case ngraph::element::Type_t::i32:
        return chainerx::Dtype::kInt32;
    case ngraph::element::Type_t::i64:
        return chainerx::Dtype::kInt64;
    case ngraph::element::Type_t::u8:
        return chainerx::Dtype::kUInt8;
    default:
        // bf16,
        // u16,
        // u32,
        // u64
        CHECK(false) << "Not supported ngraph dtype: " << type;
    }
}

chainerx::Shape GetShape(const ngraph::Shape& nshape) {
    chainerx::Shape shape;
    for (size_t d : nshape) {
        shape.push_back(d);
    }
    return shape;
}

}  // namespace

class NGraphOp::NGraphImpl {
public:
    std::shared_ptr<ngraph::Function> func;
    std::unique_ptr<ngraph::runtime::Backend> backend;
    std::shared_ptr<ngraph::runtime::Executable> handle;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result_tensors;
    std::vector<chainerx::Array> outputs;
};

#endif

void NGraphOp::InitImpl() {
#if CHAINER_COMPILER_ENABLE_NGRAPH
    impl_ = new NGraphImpl();
    std::istringstream iss(onnx);
    impl_->func = ngraph::onnx_import::import_onnx_model(iss);

    // TODO(hamaji): Make this customizable.
    const char* kBackend = "CPU";
    impl_->backend = std::move(ngraph::runtime::Backend::create(kBackend));

    impl_->handle = impl_->backend->compile(impl_->func);

    auto results = impl_->func->get_results();
    for (size_t i = 0; i < results.size(); ++i) {
        auto& tensor = results[i];
        chainerx::Dtype dtype = GetDtype(tensor->get_element_type());
        chainerx::Shape shape = GetShape(tensor->get_shape());
        chainerx::Array array = chainerx::Empty(shape, dtype);
        impl_->result_tensors.push_back(impl_->backend->create_tensor(tensor->get_element_type(), tensor->get_shape(), array.raw_data()));
        impl_->outputs.push_back(array);
    }
#endif
}

NGraphOp::~NGraphOp() {
#if CHAINER_COMPILER_ENABLE_NGRAPH
    delete impl_;
#endif
}

std::vector<chainerx::Array> NGraphOp::RunImpl(chainer_compiler::runtime::XCVMState* st, const std::vector<chainerx::Array>& orig_inputs) {
#if CHAINER_COMPILER_ENABLE_NGRAPH
    CHECK(!inputs.empty());

    size_t num_inputs = orig_inputs.size();

    // Validate inputs.
    chainerx::Array inputs[num_inputs];
    for (size_t i = 0; i < num_inputs; ++i) {
        const chainerx::Array& input = orig_inputs[i];
        if (input.IsContiguous()) {
            inputs[i] = input;
        } else {
            inputs[i] = chainerx::Copy(input);
        }
    }

    auto params = impl_->func->get_parameters();
    CHECK_EQ(params.size(), num_inputs);

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> arg_tensors(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto t = impl_->backend->create_tensor(params.at(i)->get_element_type(), params.at(i)->get_shape(), inputs[i].raw_data());
        arg_tensors.at(i) = t;
    }

    impl_->handle->call_with_validate(impl_->result_tensors, arg_tensors);

    return impl_->outputs;

#else
    CHECK(false) << "Set -DCHAINER_COMPILER_NGRAPH_DIR";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
