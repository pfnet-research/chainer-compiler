#if CHAINER_COMPILER_ENABLE_SNPE

#include <memory>
#include <unordered_map>

#include <zdl/DlContainer/IDlContainer.hpp>
#include <zdl/DlSystem/ITensorFactory.hpp>
#include <zdl/DlSystem/TensorMap.hpp>
#include <zdl/SNPE/SNPEBuilder.hpp>
#include <zdl/SNPE/SNPEFactory.hpp>

#include <runtime/chainerx_util.h>

#endif

#include <common/log.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

#if CHAINER_COMPILER_ENABLE_SNPE

namespace {

std::unordered_map<std::string, std::shared_ptr<zdl::DlContainer::IDlContainer>> dlc_cache;
std::unordered_map<std::string, std::shared_ptr<zdl::SNPE::SNPE>> snpe_cache;
zdl::DlSystem::Runtime_t snpe_runtime_type = zdl::DlSystem::Runtime_t::UNSET;

}  // namespace

#endif

SnpeDlcOp::~SnpeDlcOp() {
}

void SnpeDlcOp::InitImpl() {
}

std::vector<chainerx::Array> SnpeDlcOp::RunImpl(
        chainer_compiler::runtime::ChxVMState* st, const std::vector<chainerx::Array>& orig_inputs) {
#if CHAINER_COMPILER_ENABLE_SNPE
    // Load dlc file
    auto dlc_it = dlc_cache.find(model_data);
    if (dlc_it == dlc_cache.end()) {
        // Determine device to run
        if (snpe_runtime_type == zdl::DlSystem::Runtime_t::UNSET) {
            if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
                snpe_runtime_type = zdl::DlSystem::Runtime_t::GPU;
            } else {
                snpe_runtime_type = zdl::DlSystem::Runtime_t::CPU;
            }
        }

        std::shared_ptr<zdl::DlContainer::IDlContainer> dlc =
                zdl::DlContainer::IDlContainer::open(std::vector<uint8_t>(model_data.begin(), model_data.end()));
        dlc_it = dlc_cache.insert(std::make_pair(model_data, dlc)).first;
    }
    CHECK(dlc_it != dlc_cache.end());

    // Build SNPE object
    auto snpe_it = snpe_cache.find(model_data);
    if (snpe_it == snpe_cache.end()) {
        const std::shared_ptr<zdl::DlContainer::IDlContainer>& dlc = dlc_it->second;
        zdl::SNPE::SNPEBuilder snpeBuilder(dlc.get());
        std::shared_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputLayers({}).setRuntimeProcessor(snpe_runtime_type).build();
        snpe_it = snpe_cache.insert(std::make_pair(model_data, snpe)).first;
    }
    CHECK(snpe_it != snpe_cache.end());

    zdl::SNPE::SNPE& snpe = *snpe_it->second;
    auto& tensor_factory = zdl::SNPE::SNPEFactory::getTensorFactory();

    // Build input tensors
    zdl::DlSystem::TensorMap input_map;
    std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> input_tensors;
    for (size_t i = 0; i < input_names.size(); ++i) {
        const chainerx::Array& orig_input = orig_inputs[i];
        std::vector<zdl::DlSystem::Dimension> input_shape(orig_input.shape().begin(), orig_input.shape().end());
        auto input =
                tensor_factory.createTensor(input_shape, reinterpret_cast<const uint8_t*>(orig_input.raw_data()), orig_input.GetNBytes());
        input_map.add(input_names[i].c_str(), input.get());
        input_tensors.push_back(std::move(input));
    }

    // Build output tensor map
    zdl::DlSystem::TensorMap output_map;

    snpe.execute(input_map, output_map);

    std::vector<chainerx::Array> ret;
    for (auto name : output_map.getTensorNames()) {
        const zdl::DlSystem::ITensor* output_tensor = output_map.getTensor(name);
        CHECK(output_tensor);
        const auto& orig_shape = output_tensor->getShape();
        chainerx::Shape shape(orig_shape.getDimensions(), orig_shape.getDimensions() + orig_shape.rank());
        ret.push_back(MakeArray(chainerx::Dtype::kFloat32, shape, output_tensor->cbegin().dataPointer()));
    }

    return ret;
#else
    CHECK(false) << "Set -DCHAINER_COMPILER_SNPE_INCLUDE_DIR and -DCHAINER_COMPILER_SNPE_LIBRARY_DIR";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
