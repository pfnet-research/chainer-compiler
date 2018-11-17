#include <gtest/gtest.h>

#include <chainerx/context.h>

#include <common/log.h>
#include <compiler/evaluator.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/value.h>

namespace oniku {
namespace {

TEST(EvaluatorTest, Eval) {
    Value dummy_for_test("test");
    Graph graph("test");
    GraphBuilder gb(&graph, "test", &dummy_for_test);
    Value* a = gb.Const(Type(Dtype::kInt32, {2}), {3, 10});
    Value* b = gb.Const(Type(Dtype::kInt32, {2}), {7, 32});
    Value* r = gb.Op(Node::kAdd, {a, b});

    chainerx::Context ctx;
    chainerx::SetGlobalDefaultContext(&ctx);
    const std::vector<Node*> nodes = {a->producer(), b->producer(), r->producer()};
    std::vector<std::unique_ptr<Tensor>> outputs;
    Eval(nodes, {r}, &outputs);
    ASSERT_EQ(1UL, outputs.size());
    EXPECT_EQ(Dtype::kInt32, outputs[0]->dtype());
    ASSERT_EQ(1, outputs[0]->dims().size());
    EXPECT_EQ(2, outputs[0]->dims()[0]);
    EXPECT_EQ(10, outputs[0]->Get<int>(0));
    EXPECT_EQ(42, outputs[0]->Get<int>(1));
}

}  // namespace
}  // namespace oniku
