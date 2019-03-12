#include <gtest/gtest.h>

#include <compiler/flops.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/type.h>

namespace chainer_compiler {
namespace {

TEST(FlopsTest, Conv) {
    Graph graph("test");
    Value* x = graph.AddInputValue("x", Type(Dtype::kFloat32, {2, 6, 7, 8}));
    Value* w = graph.AddInputValue("w", Type(Dtype::kFloat32, {4, 6, 3, 2}));
    Value* y = nullptr;
    {
        GraphBuilder gb(&graph, "test", x);
        y = gb.Op(Node::kConv, {x, w});
    }
    int num_unknown_ops = 0;
    EXPECT_EQ(10080, CalculateFlops(*y->producer(), &num_unknown_ops));
}

}  // namespace
}  // namespace chainer_compiler
