#include <iostream>

#include <gtest/gtest.h>

#include <xchainer/array.h>
#include <xchainer/context.h>
#include <xchainer/routines/creation.h>
#include <xchainer/testing/array.h>
#include <xchainer/testing/array_check.h>

#include <runtime/xcvm.h>
#include <runtime/xcvm.pb.h>
#include <runtime/xcvm_proto_util.h>

namespace oniku {
namespace runtime {
namespace {

TEST(XCVMTest, Run) {
    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);

    XCProgramProto program;
    *program.add_instructions() = MakeInOp(0, "in1");
    *program.add_instructions() = MakeInOp(1, "in2");
    *program.add_instructions() = MakeAddOp(2, 0, 1);
    *program.add_instructions() = MakeOutOp("out", 2);
    // std::cerr << program.DebugString() << std::endl;

    XCVM xcvm(program);
    InOuts inputs;
    inputs["in1"] = xchainer::Eye(2, nonstd::nullopt, nonstd::nullopt, xchainer::Dtype::kFloat32);
    inputs["in2"] = xchainer::OnesLike(inputs["in1"]);
    InOuts outputs = xcvm.Run(inputs, false);
    ASSERT_EQ(1, outputs.count("out"));
    xchainer::Array e = xchainer::testing::BuildArray({2, 2}).WithData<float>({2, 1, 1, 2});
    xchainer::testing::ExpectEqual(e, outputs["out"]);
}

}  // namespace
}  // namespace runtime
}  // namespace oniku
