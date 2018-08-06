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

TEST(ModelTest, DumpSimpleONNX) {
    std::string path = (std::string(kONNXTestDataDir) + "/simple/test_single_relu_model/model.onnx");
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(path));
    Model model(xmodel);
    onnx::ModelProto xmodel2;
    model.ToONNX(&xmodel2);
    ASSERT_EQ(xmodel.DebugString(), xmodel2.DebugString());
}

}  // namespace
}  // namespace oniku
