#if CHAINER_COMPILER_ENABLE_DLDT

#include <map>
#include <sstream>

#include <nonstd/optional.hpp>

#include <chainerx/routines/creation.h>

#include <inference_engine.hpp>

#include <common/log.h>

#else

#include <common/log.h>

#endif

#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

#if CHAINER_COMPILER_ENABLE_DLDT

namespace {

#if 0

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

#endif

}  // namespace

class DldtOp::DldtImpl {
public:
    explicit DldtImpl(const std::string& device)
        : plugin(InferenceEngine::PluginDispatcher().getSuitablePlugin(InferenceEngine::TargetDeviceInfo::fromStr(device))) {
    }
    InferenceEngine::InferencePlugin plugin;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executable_network;
    std::vector<chainerx::Array> output_arrays;
};

#endif

void DldtOp::InitImpl() {
#if CHAINER_COMPILER_ENABLE_DLDT
    impl_ = new DldtImpl(device);

    std::cerr << "Loading dldt model from " << model_path << std::endl;
    InferenceEngine::CNNNetReader network_reader;
    network_reader.ReadNetwork(model_path + ".xml");
    network_reader.ReadWeights(model_path + ".bin");
    // network_reader.getNetwork().setBatchSize(1);
    impl_->network = network_reader.getNetwork();

    impl_->executable_network = impl_->plugin.LoadNetwork(impl_->network, {});

    CHECK_EQ(inst_.output_types().size(), inst_.output_names().size());
    for (const XCTypeProto& type : inst_.output_types()) {
        CHECK_NE(type.dtype(), 0);
        chainerx::Shape shape(type.shape().begin(), type.shape().end());
        chainerx::Dtype dtype = static_cast<chainerx::Dtype>(type.dtype());
        chainerx::Array array = chainerx::Empty(shape, dtype);
        impl_->output_arrays.push_back(array);
    }
#endif
}

DldtOp::~DldtOp() {
#if CHAINER_COMPILER_ENABLE_DLDT
    delete impl_;
#endif
}

std::vector<chainerx::Array> DldtOp::RunImpl(chainer_compiler::runtime::ChxVMState* st, const std::vector<chainerx::Array>& orig_inputs) {
#if CHAINER_COMPILER_ENABLE_DLDT
    CHECK(!inputs.empty());

    size_t num_inputs = orig_inputs.size();

    CHECK_EQ(num_inputs, impl_->network.getInputsInfo().size());
#if 0
    // set layout and precision?
    {
        InputInfo::Ptr input_info = network.getInputsInfo().begin();
        for (int i = 0; i < num_inputs; ++i) {
        }
    }
#endif

    // Validate inputs.
    chainerx::Array inputs[num_inputs];
    for (size_t i = 0; i < num_inputs; ++i) {
        const chainerx::Array& input = orig_inputs[i];
        inputs[i] = chainerx::AsContiguous(input);
    }

    InferenceEngine::InferRequest infer_request = impl_->executable_network.CreateInferRequest();

    {
        auto inputs_info = impl_->network.getInputsInfo();
        auto input_iter = inputs_info.begin();
        for (int i = 0; i < num_inputs; ++i, ++input_iter) {
            CHECK(input_iter != inputs_info.end());
            InferenceEngine::Blob::Ptr input = infer_request.GetBlob(input_iter->first);
            auto input_data = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            memcpy(input_data, inputs[i].raw_data(), inputs[i].GetNBytes());
        }
    }

    infer_request.Infer();

    // We assume output values are alphabetically sorted in ONNX.
    size_t i = 0;
    for (auto& p : impl_->network.getOutputsInfo()) {
        InferenceEngine::Blob::Ptr output = infer_request.GetBlob(p.first);
        const chainerx::Array& output_array = impl_->output_arrays[i];
        memcpy(output_array.raw_data(), output->buffer(), output_array.GetNBytes());
        ++i;
    }

    return impl_->output_arrays;

#else
    CHECK(false) << "Set -DCHAINER_COMPILER_DLDT_DIR";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
