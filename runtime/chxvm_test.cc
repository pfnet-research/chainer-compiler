#include <iostream>

#include <gtest/gtest.h>

#include <chainerx/array.h>
#include <chainerx/numeric.h>
#include <chainerx/routines/creation.h>
#include <chainerx/testing/array.h>
#include <chainerx/testing/context_session.h>

#include <compiler/chxvm/chxvm_value.h>
#include <compiler/gen_chxvm_codegen.h>
#include <runtime/chxvm.h>
#include <runtime/chxvm.pb.h>
#include <runtime/chxvm_var.h>

namespace chainer_compiler {
namespace runtime {
namespace {

TEST(ChxVMTest, Run) {
    chainerx::testing::ContextSession sess;

    XCProgramProto program;
    chxvm::AddInOp(&program, chxvm::ChxVMValue(0), "in1");
    chxvm::AddInOp(&program, chxvm::ChxVMValue(1), "in2");
    chxvm::AddAddOp(&program, chxvm::ChxVMValue(2), 0, 1);
    chxvm::AddOutOp(&program, "out", 2);
    // std::cerr << program.DebugString() << std::endl;

    ChxVM chxvm(program);
    InOuts inputs;
    chainerx::Array in1 = chainerx::Eye(2, nonstd::nullopt, nonstd::nullopt, chainerx::Dtype::kFloat32);
    inputs.emplace("in1", std::shared_ptr<ChxVMVar>(new ChxVMVar(in1)));
    inputs.emplace("in2", std::shared_ptr<ChxVMVar>(new ChxVMVar(chainerx::OnesLike(in1))));
    InOuts outputs = chxvm.Run(inputs, ChxVMOptions());
    ASSERT_EQ(1, outputs.count("out"));
    chainerx::Array e = chainerx::testing::BuildArray({2, 2}).WithData<float>({2, 1, 1, 2});
    // TODO(hamaji): Use EXPECT_ARRAY_EQ after fixing namespace?
    EXPECT_TRUE(chainerx::AllClose(e, outputs["out"]->GetArray(), 0, 0));
}

}  // namespace
}  // namespace runtime
}  // namespace chainer_compiler
