#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <compiler/onnx.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/chxvm/emitter.h>
#include <compiler/model.h>
#include <compiler/passes.h>
#include <runtime/chxvm.pb.h>

namespace chainer_compiler {
namespace {

const char* kONNXTestDataDir = "third_party/onnx/onnx/backend/test/data";

TEST(ChxVMTest, Compile) {
    std::string test_path = std::string(kONNXTestDataDir) + "/node/test_add/";
    std::string model_path = test_path + "model.onnx";
    onnx::ModelProto xmodel(LoadLargeProto<onnx::ModelProto>(model_path));
    Model model(xmodel);
    RunDefaultPasses(&model);

    std::ostringstream oss;
    chxvm::Emit(model, oss);

    runtime::ChxVMProgramProto program;
    program.ParseFromString(oss.str());
    // std::cerr << program.DebugString() << std::endl;

    ASSERT_EQ(7, program.instructions_size());
    ASSERT_EQ(runtime::ChxVMInstructionProto::In, program.instructions(0).op());
    ASSERT_EQ(runtime::ChxVMInstructionProto::In, program.instructions(1).op());
    ASSERT_EQ(runtime::ChxVMInstructionProto::Add, program.instructions(2).op());
    ASSERT_EQ(runtime::ChxVMInstructionProto::Free, program.instructions(3).op());
    ASSERT_EQ(runtime::ChxVMInstructionProto::Free, program.instructions(4).op());
    ASSERT_EQ(runtime::ChxVMInstructionProto::Out, program.instructions(5).op());
    ASSERT_EQ(runtime::ChxVMInstructionProto::Free, program.instructions(6).op());
}

}  // namespace
}  // namespace chainer_compiler
