#if CHAINER_COMPILER_ENABLE_DLDT

#include <map>
#include <sstream>

#include <absl/types/optional.h>

#include <chainerx/routines/creation.h>

#include <inference_engine.hpp>

#endif

#include <common/log.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

#if CHAINER_COMPILER_ENABLE_DLDT

namespace {

chainerx::Dtype GetDtype(InferenceEngine::Precision::ePrecision type) {
    switch (type) {
        case InferenceEngine::Precision::FP32:
            return chainerx::Dtype::kFloat32;
        case InferenceEngine::Precision::FP16:
            return chainerx::Dtype::kFloat16;
        case InferenceEngine::Precision::I16:
            return chainerx::Dtype::kInt16;
        case InferenceEngine::Precision::U8:
            return chainerx::Dtype::kUInt8;
        case InferenceEngine::Precision::I8:
            return chainerx::Dtype::kInt8;
        case InferenceEngine::Precision::I32:
            return chainerx::Dtype::kInt32;
        case InferenceEngine::Precision::BIN:
            return chainerx::Dtype::kBool;
        default:
            // UNSPECIFIED,
            // MIXED,
            // Q78,
            // uU16,
            CHECK(false) << "Not supported dldt dtype: " << type;
    }
}

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

    for (const std::string& name : output_names) {
        impl_->network.addOutput(name, 0);
    }

    impl_->executable_network = impl_->plugin.LoadNetwork(impl_->network, {});

    CHECK_EQ(inst_.output_types().size(), inst_.output_names().size());
    CHECK_EQ(output_names.size(), inst_.output_names().size());
    InferenceEngine::OutputsDataMap outputs_info(impl_->network.getOutputsInfo());
    for (size_t i = 0; i < output_names.size(); ++i) {
        const std::string& name = output_names[i];
        const ChxVMTypeProto& type = inst_.output_types(i);
        CHECK_NE(type.dtype(), 0);

        auto found = outputs_info.find(name);
        CHECK(found != outputs_info.end()) << name;
        const InferenceEngine::DataPtr& data = found->second;
        chainerx::Dtype dtype = GetDtype(data->precision);
        chainerx::Shape shape(type.shape().begin(), type.shape().end());
        CHECK_EQ(chainerx::Shape(data->dims.rbegin(), data->dims.rend()), shape);
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

    std::vector<chainerx::Array> output_arrays;
    for (size_t i = 0; i < output_names.size(); ++i) {
        const std::string& output_name = output_names[i];
        InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);
        const chainerx::Array& output_array = impl_->output_arrays[i];
        memcpy(output_array.raw_data(), output->buffer(), output_array.GetNBytes());
        chainerx::Dtype dtype = static_cast<chainerx::Dtype>(inst_.output_types(i).dtype());
        if (output_array.dtype() != dtype) {
            output_arrays.push_back(output_array.AsType(dtype));
        } else {
            output_arrays.push_back(output_array);
        }
    }

    return output_arrays;

#else
    CHECK(false) << "Set -DCHAINER_COMPILER_DLDT_DIR";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
