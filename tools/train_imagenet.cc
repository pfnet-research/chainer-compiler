#include <cuda.h>
#include <cuda_runtime.h>

#include <onnx/onnx.pb.h>

#include <xchainer/array.h>
#include <xchainer/context.h>
#include <xchainer/routines/creation.h>
#include <xchainer/routines/manipulation.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <compiler/tensor.h>
#include <compiler/value.h>
#include <compiler/xcvm_emitter.h>
#include <feeder/imagenet_iterator.h>
#include <runtime/xchainer.h>
#include <runtime/xcvm.h>
#include <tools/cmdline.h>

namespace oniku {
namespace runtime {
namespace {

bool g_quiet;

#define LOG()                                   \
    if (!g_quiet) std::cerr

xchainer::Dtype XChainerTypeFromONNX(onnx::TensorProto::DataType xtype) {
    switch (xtype) {
        case onnx::TensorProto::BOOL:
            return xchainer::Dtype::kBool;
        case onnx::TensorProto::INT8:
            return xchainer::Dtype::kInt8;
        case onnx::TensorProto::INT16:
            return xchainer::Dtype::kInt16;
        case onnx::TensorProto::INT32:
            return xchainer::Dtype::kInt32;
        case onnx::TensorProto::INT64:
            return xchainer::Dtype::kInt64;
        case onnx::TensorProto::UINT8:
            return xchainer::Dtype::kUInt8;
        case onnx::TensorProto::FLOAT:
            return xchainer::Dtype::kFloat32;
        case onnx::TensorProto::DOUBLE:
            return xchainer::Dtype::kFloat64;
        default:
            CHECK(false) << "Unsupported ONNX data type: " << xtype;
    }
}

void RunMain(int argc, char** argv) {
    cmdline::parser args;
    args.add<int>("batchsize", 'B', "Batch size", false, 32);
    args.add<std::string>("device", 'd', "xChainer device to be used", false);
    args.add("trace", 't', "Tracing mode");
    args.add("verbose", 'v', "Verbose mode");
    args.add("quiet", 'q', "Quiet mode");
    args.parse_check(argc, argv);
    if (args.rest().size() != 3) {
        std::cerr << args.usage() << std::endl;
        QFAIL() << "Usage: " << argv[0] << " <onnx> <train.txt> <mean.bin>";
    }

    g_quiet = args.exist("quiet");
    int batch_size = args.get<int>("batchsize");

    LOG() << "Initializing xChainer..." << std::endl;
    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);
    const std::string device = args.get<std::string>("device");
    size_t initial_free_bytes, param_bytes = static_cast<size_t>(-1);
    if (!device.empty()) {
        CHECK_EQ(cudaSuccess, cudaMemGetInfo(&initial_free_bytes, nullptr));
        xchainer::SetDefaultDevice(&xchainer::GetDefaultContext().GetDevice(device));
    }

    LOG() << "Constructing model..." << std::endl;
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(args.rest()[0]));
    Model model(xmodel);
    RunDefaultPasses(model.mutable_graph(), true  /* gen_backprop */);

    LOG() << "Loading data..." << std::endl;

    InOuts params;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (const Value* input : model.graph().input_values()) {
        if (const Tensor* initializer = input->initializer()) {
            xchainer::Dtype dtype = XChainerTypeFromONNX(initializer->dtype().ToONNX());
            xchainer::Shape shape(initializer->dims());
            const void* data = initializer->GetRawData();
            xchainer::Array tensor(MakeArray(dtype, shape, data));
            CHECK(params.emplace(initializer->name(), tensor).second) << "Duplicate input tensor: " << initializer->name();
        } else {
            input_names.push_back(input->name());
        }
    }
    xchainer::Array batch_size_array = MakeArray(xchainer::Dtype::kFloat32, {}, &batch_size).ToDevice(xchainer::GetDefaultContext().GetDevice(device));
    // TODO(hamaji): This is not actually a batch size. Rename it.
    batch_size_array *= 1000;

    LOG() << "Generate code..." << std::endl;
    XCProgramProto xcvm_prog;
    xcvm::Emit(model, &xcvm_prog);
    XCVM xcvm(xcvm_prog);
    XCVMOptions xcvm_opts;
    xcvm_opts.trace_level = args.exist("verbose") ? 2 : args.exist("trace") ? 1 : 0;
    xcvm_opts.is_training = true;

    // TODO(hamaji): Stop using the fixed width/height.
    const int kHeight = 227;
    const int kWidth = 227;
    const std::vector<float>& mean = LoadMean(args.rest()[2], kHeight, kWidth);
    ImageNetIterator train_iter(args.rest()[1], 3, batch_size, mean, kHeight, kWidth);
    train_iter.Start();

    LOG() << "Start training!" << std::endl;
    int iter_count = 0;
    for (;; ++iter_count) {
        std::vector<xchainer::Array> data = train_iter.GetNext();
        if (data.empty())
            break;

        InOuts inputs(params);
        inputs["input"] = data[0].ToDevice(xchainer::GetDefaultContext().GetDevice(device));
        xchainer::Array labels = data[1].ToDevice(xchainer::GetDefaultContext().GetDevice(device)).AsType(xchainer::Dtype::kInt64);
        xchainer::Array onehot = xchainer::Eye(1000, nonstd::nullopt, nonstd::nullopt, xchainer::Dtype::kFloat32).Take(labels, 0);
        inputs["onehot"] = onehot;
        inputs["batch_size"] = batch_size_array;
        InOuts outputs(xcvm.Run(inputs, xcvm_opts));

        double loss = static_cast<double>(xchainer::AsScalar(outputs["loss"]));

        std::cout << train_iter.GetStatus() << " loss=" << loss << std::endl;
    }

    train_iter.Terminate();
}

}  // namespace
}  // namespace runtime
}  // namespace oniku

int main(int argc, char** argv) {
    oniku::runtime::RunMain(argc, argv);
}
