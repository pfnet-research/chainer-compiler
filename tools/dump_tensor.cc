// Dump an ONNX TensorProto

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

#include <onnx/onnx.pb.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/tensor.h>

int main(int argc, const char** argv) {
    if (argc <= 1) {
        QFAIL() << "Usage: " << argv[0] << " <onnx>";
    }

    onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(argv[1]));
    oniku::Tensor tensor(xtensor);
    onnx::TensorProto xtensor_normalized;
    tensor.ToONNX(&xtensor_normalized);
    std::cout << xtensor_normalized.DebugString();
}
