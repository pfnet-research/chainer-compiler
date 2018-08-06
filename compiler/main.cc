#include <iostream>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/model.h>
#include <compiler/xchainer.h>

int main(int argc, const char** argv) {
    if (argc <= 1) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>";
    }

    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(argv[1]));
    oniku::Model model(xmodel);
    oniku::xchainer::Emit(model, std::cout);
}
