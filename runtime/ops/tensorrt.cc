#if CHAINER_COMPILER_ENABLE_TENSORRT

#include <sstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chainerx/routines/creation.h>

#endif

#include <common/log.h>
#include <runtime/gen_chxvm_ops.h>

namespace chainer_compiler {
namespace runtime {

#if CHAINER_COMPILER_ENABLE_TENSORRT

namespace {

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
};

#endif

void TensorRTOp::InitImpl() {
#if CHAINER_COMPILER_ENABLE_TENSORRT
    impl_ = new TensorRTImpl();
    std::istringstream iss(onnx);

    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(impl_->logger));

    //nvonnxparser::createParser();
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
        inputs[i] = chainerx::AsContiguous(input);
    }

    CHECK(false) << "TODO(hamaji): Implement";
#else
    CHECK(false) << "Set -DCHAINER_COMPILER_ENABLE_TENSORRT";
#endif
}

}  // namespace runtime
}  // namespace chainer_compiler
