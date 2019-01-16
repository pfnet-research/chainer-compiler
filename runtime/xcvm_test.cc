#include <iostream>

#include <gtest/gtest.h>

#include <chainerx/array.h>
#include <chainerx/context.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>
#include <chainerx/testing/array.h>

#include <compiler/gen_xcvm_codegen.h>
#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_var.h>

namespace chainer_compiler {
namespace runtime {
namespace {

TEST(XCVMTest, Run) {
    chainerx::Context ctx;
    chainerx::SetGlobalDefaultContext(&ctx);

    XCProgramProto program;
    xcvm::AddInOp(&program, 0, "in1");
    xcvm::AddInOp(&program, 1, "in2");
    xcvm::AddAddOp(&program, 2, 0, 1);
    xcvm::AddOutOp(&program, "out", 2);
    // std::cerr << program.DebugString() << std::endl;

    XCVM xcvm(program);
    InOuts inputs;
    chainerx::Array in1 = chainerx::Eye(2, nonstd::nullopt, nonstd::nullopt, chainerx::Dtype::kFloat32);
    inputs.emplace("in1", std::shared_ptr<XCVMVar>(new XCVMVar(in1)));
    inputs.emplace("in2", std::shared_ptr<XCVMVar>(new XCVMVar(chainerx::OnesLike(in1))));
    InOuts outputs = xcvm.Run(inputs, XCVMOptions());
    ASSERT_EQ(1, outputs.count("out"));
    chainerx::Array e = chainerx::testing::BuildArray({2, 2}).WithData<float>({2, 1, 1, 2});
    // TODO(hamaji): Use EXPECT_ARRAY_EQ after fixing namespace?
    EXPECT_TRUE(chainerx::AllClose(e, outputs["out"]->GetArray(), 0, 0));
}

}  // namespace
}  // namespace runtime
}  // namespace chainer_compiler
