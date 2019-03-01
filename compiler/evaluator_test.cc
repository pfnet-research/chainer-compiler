#include <gtest/gtest.h>

#include <chainerx/context.h>

#include <common/log.h>
#include <compiler/evaluator.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/value.h>

namespace chainer_compiler {
namespace {

TEST(EvaluatorTest, Eval) {
    chainerx::Context ctx;
    chainerx::ContextScope ctx_scope(ctx);

    Value dummy_for_test("test");
    Graph graph("test");
    GraphBuilder gb(&graph, "test", &dummy_for_test);
    Value* a = gb.Const(Type(Dtype::kInt32, {2}), {3, 10});
    Value* b = gb.Const(Type(Dtype::kInt32, {2}), {7, 32});
    Value* r = gb.Op(Node::kAdd, {a, b});

    const std::vector<Node*> nodes = {a->producer(), b->producer(), r->producer()};
    std::vector<std::unique_ptr<EvaluatedValue>> outputs;
    Eval(nodes, {r}, &outputs);
    ASSERT_EQ(1UL, outputs.size());
    ASSERT_TRUE(outputs[0]->is_tensor());
    std::unique_ptr<Tensor> out(outputs[0]->ReleaseTensor());
    EXPECT_EQ(Dtype::kInt32, out->dtype());
    ASSERT_EQ(1, out->dims().size());
    EXPECT_EQ(2, out->dims()[0]);
    EXPECT_EQ(10, out->Get<int>(0));
    EXPECT_EQ(42, out->Get<int>(1));
}

TEST(EvaluatorTest, EvalWithFeeds) {
    chainerx::Context ctx;
    chainerx::ContextScope ctx_scope(ctx);

    Value dummy_for_test("test");
    Graph graph("test");
    GraphBuilder gb(&graph, "test", &dummy_for_test);
    Value* a = gb.Const(Type(Dtype::kInt32, {2}), {0, 0});
    Value* b = gb.Const(Type(Dtype::kInt32, {2}), {0, 0});
    Value* r = gb.Op(Node::kAdd, {a, b});

    const std::vector<Node*> nodes = {r->producer()};
    std::vector<std::pair<Value*, Tensor*>> feeds;
    feeds.emplace_back(a, new Tensor("a", Dtype::kInt32, {2}, {3, 10}));
    feeds.emplace_back(b, new Tensor("b", Dtype::kInt32, {2}, {7, 32}));
    std::vector<std::unique_ptr<EvaluatedValue>> outputs;
    Eval(nodes, feeds, {r}, &outputs);
    ASSERT_EQ(1UL, outputs.size());
    ASSERT_TRUE(outputs[0]->is_tensor());
    std::unique_ptr<Tensor> out(outputs[0]->ReleaseTensor());
    EXPECT_EQ(Dtype::kInt32, out->dtype());
    ASSERT_EQ(1, out->dims().size());
    EXPECT_EQ(2, out->dims()[0]);
    EXPECT_EQ(10, out->Get<int>(0));
    EXPECT_EQ(42, out->Get<int>(1));
    for (const auto& p : feeds) {
        delete p.second;
    }
}

}  // namespace
}  // namespace chainer_compiler
