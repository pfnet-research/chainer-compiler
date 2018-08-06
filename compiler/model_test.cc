#include <string>

#include <fstream>

#include <gtest/gtest.h>

#include <common/protoutil.h>
#include <compiler/model.h>

namespace oniku {
namespace {

const char* kONNXTestDataDir = "../onnx/onnx/backend/test/data";

TEST(ModelTest, ReadSimpleONNX) {
    std::string path = (std::string(kONNXTestDataDir) + "/simple/test_single_relu_model/model.onnx");
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);
}

}  // namespace
}  // namespace oniku
