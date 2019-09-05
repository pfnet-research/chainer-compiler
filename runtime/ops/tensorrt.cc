#if CHAINER_COMPILER_ENABLE_TENSORRT

#include <sstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chainerx/routines/creation.h>

#include <runtime/chainerx_util.h>

#endif

#include <common/log.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

#if CHAINER_COMPILER_ENABLE_TENSORRT

namespace {

constexpr int kBatchSize = 1;

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

class Logger : public nvinfer1::ILogger {
public:
    ~Logger() override {
    }

    void log(Severity severity, const char* msg) override {
        std::cerr << msg << std::endl;
    }
};

}  // namespace

class TensorRTOp::TensorRTImpl {
public:
    Logger logger;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
};

#endif

void TensorRTOp::InitImpl() {
#if CHAINER_COMPILER_ENABLE_TENSORRT
    impl_ = new TensorRTImpl();
    std::istringstream iss(onnx);

    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(impl_->logger));
    CHECK(builder);
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    CHECK(network);
    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, impl_->logger));

    if (!parser->parse(iss.str().c_str(), iss.str().size())) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            auto e = parser->getError(i);
            std::cerr << e->desc() << std::endl;
        }
        CHECK(false);
    }

    builder->setMaxBatchSize(kBatchSize);
    builder->setMaxWorkspaceSize(128 * 1000 * 1000);
    // builder->setFp16Mode(false);

    impl_->engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), InferDeleter());
#endif
}

TensorRTOp::~TensorRTOp() {
#if CHAINER_COMPILER_ENABLE_TENSORRT
    delete impl_;
#endif
}

std::vector<chainerx::Array> TensorRTOp::RunImpl(chainer_compiler::runtime::ChxVMState* st, const std::vector<chainerx::Array>& orig_inputs) {
#if CHAINER_COMPILER_ENABLE_TENSORRT
    size_t num_inputs = orig_inputs.size();

    // Validate inputs.
    chainerx::Array inputs[num_inputs];
    for (size_t i = 0; i < num_inputs; ++i) {
        const chainerx::Array& input = orig_inputs[i];
        CHECK_EQ(input.shape()[0], kBatchSize);
        inputs[i] = chainerx::AsContiguous(input);
    }

    auto context = UniquePtr<nvinfer1::IExecutionContext>(impl_->engine->createExecutionContext());
    CHECK(context);

    std::vector<void*> bindings;
    for (const chainerx::Array& input : inputs) {
        bindings.push_back(RawStartPtr(input));
    }

    const bool status = context->execute(kBatchSize, &bindings[0]);
    CHECK(status);

    CHECK(false) << "TODO";

#else
    CHECK(false) << "Set -DCHAINER_COMPILER_ENABLE_TENSORRT";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
