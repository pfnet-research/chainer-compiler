#include <iostream>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/model.h>
#include <compiler/scheduler.h>
#include <compiler/xchainer.h>

namespace oniku {

void RunMain(int argc, const char** argv) {
    if (argc <= 1) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>";
    }

    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(argv[1]));
    Model model(xmodel);
    ScheduleComputation(model.graph());
    xchainer::Emit(model, std::cout);
}

}  // namespace oniku

int main(int argc, const char** argv) {
    oniku::RunMain(argc, argv);
}
