#if CHAINER_COMPILER_ENABLE_DLDT

#include <sstream>

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
};

#endif

void DldtOp::InitImpl() {
#if CHAINER_COMPILER_ENABLE_DLDT
    impl_ = new DldtImpl();
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

    // Validate inputs.
    chainerx::Array inputs[num_inputs];
    for (size_t i = 0; i < num_inputs; ++i) {
        const chainerx::Array& input = orig_inputs[i];
        inputs[i] = chainerx::AsContiguous(input);
    }

    using namespace InferenceEngine;

    InferencePlugin plugin(PluginDispatcher().getSuitablePlugin(TargetDevice::eCPU));

    CNNNetReader network_reader;
    network_reader.ReadNetwork(model_path + ".xml");
    network_reader.ReadWeights(model_path + ".bin");
    // network_reader.getNetwork().setBatchSize(1);
    CNNNetwork network = network_reader.getNetwork();

    CHECK_EQ(num_inputs, network.getInputsInfo().size());
#if 0
    // set layout and precision?
    {
        InputInfo::Ptr input_info = network.getInputsInfo().begin();
        for (int i = 0; i < num_inputs; ++i) {
        }
    }
#endif

    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});

    InferRequest infer_request = executable_network.CreateInferRequest();

    {
        auto inputs_info = network.getInputsInfo();
        auto input_iter = inputs_info.begin();
        fprintf(stderr, "%d %d\n", (int)num_inputs, (int)inputs_info.size());
        for (int i = 0; i < num_inputs; ++i, ++input_iter) {
            CHECK(input_iter != inputs_info.end());
            Blob::Ptr input = infer_request.GetBlob(input_iter->first);
            auto input_data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            memcpy(input_data, inputs[i].raw_data(), inputs[i].GetNBytes());
        }
    }

    infer_request.Infer();

    Blob::Ptr output = infer_request.GetBlob(network.getOutputsInfo().begin()->first);

    chainerx::Shape output_shape;
    // TODO(hamaji): Fix the shape.
    // for (int64_t d : output->dims()) {
    for (int i = output->dims().size(); --i >= 0;) {
        output_shape.push_back(output->dims()[i]);
    }

    chainerx::Array output_array = chainerx::Empty(output_shape, chainerx::Dtype::kFloat32);
    memcpy(output_array.raw_data(), output->buffer(), output_array.GetNBytes());

    return {output_array};

#else
    CHECK(false) << "Set -DCHAINER_COMPILER_DLDT_DIR";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
