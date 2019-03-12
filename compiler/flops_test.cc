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

    auto make_conv = [&](int group, int dilation) {
        GraphBuilder gb(&graph, "test", x);
        Value* y = gb.Op(Node::kConv, {x, w});
        y->producer()->set_group(group)->set_dilations({dilation, dilation});
        return y->producer();
    };

    int num_unknown_ops = 0;
    EXPECT_EQ(10080, CalculateFlops(*make_conv(1, 1), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ(5040, CalculateFlops(*make_conv(2, 1), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);

    EXPECT_EQ(5184, CalculateFlops(*make_conv(1, 2), &num_unknown_ops));
    EXPECT_EQ(0, num_unknown_ops);
}

}  // namespace
}  // namespace chainer_compiler
