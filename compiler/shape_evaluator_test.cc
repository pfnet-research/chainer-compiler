#include <gtest/gtest.h>

#include <chainerx/testing/context_session.h>

#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/shape_evaluator.h>
#include <compiler/value.h>

namespace chainer_compiler {
namespace {

using chainerx::testing::array_detail::ArrayBuilder;

TEST(ShapeEvaluatorTest, EvaluateShapes) {
    chainerx::testing::ContextSession sess;

    Value dummy_for_test("test");
    Graph graph({}, "test");
    GraphBuilder gb(&graph, "test", &dummy_for_test);
    Value* a = gb.Const(ArrayBuilder({2, 1, 1}).WithData<int32_t>({0, 0}).Build());
    Value* b = gb.Const(ArrayBuilder({1, 1, 3}).WithData<int32_t>({0, 0, 0}).Build());
    Value* r = gb.Op(Node::kAdd, {a, b});
    ASSERT_TRUE(a->type().HasKnownShape());
    ASSERT_TRUE(b->type().HasKnownShape());
    ASSERT_FALSE(r->type().HasKnownShape());

    DoEvaluateShape(r->producer());

    ASSERT_TRUE(r->type().HasKnownShape());
    EXPECT_EQ(Dtype::kInt32, r->type().dtype());
    std::vector<int64_t> expected_dims = {2, 1, 3};
    EXPECT_EQ(expected_dims, r->type().dims());
}

}  // namespace
}  // namespace chainer_compiler
