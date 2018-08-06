// Dump an ONNX TensorProto

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <common/protoutil.h>

int main(int argc, const char** argv) {
    if (argc <= 1) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>";
    }

    onnx::TensorProto tensor(LoadLargeProto<onnx::TensorProto>(argv[1]));
    std::cout << tensor.DebugString();
}
