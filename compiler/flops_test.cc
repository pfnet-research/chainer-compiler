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

TEST(FlopsTest, ConvGradWeight) {
    Graph graph("test");
    Value* w = graph.AddInputValue("w", Type(Dtype::kFloat32, {4, 6, 3, 2}));
    Value* x = graph.AddInputValue("x", Type(Dtype::kFloat32, {2, 6, 7, 8}));
    Value* gy = graph.AddInputValue("gy", Type(Dtype::kFloat32, {2, 10, 7, 8}));

    auto make_conv_grad_weight = [&](int group) {
        GraphBuilder gb(&graph, "test", x);
        Value* y = gb.Op(Node::kChainerConvGradWeight, {w, x, gy});
        y->producer()->set_group(group);
        return y->producer();
    };

    EXPECT_EQ(2 * 7 * 8 * 6 * 10, CalculateFlops(*make_conv_grad_weight(1)));
}

TEST(FlopsTest, Gemm) {
    Graph graph("test");
    Value* a = graph.AddInputValue("a", Type(Dtype::kFloat32, {2, 6}));
    Value* b = graph.AddInputValue("b", Type(Dtype::kFloat32, {6, 2}));
    Value* c = graph.AddInputValue("c", Type(Dtype::kFloat32, {4, 2}));

    auto make_gemm = [&](double alpha, double beta) {
        GraphBuilder gb(&graph, "test", a);
        Value* n = gb.Op(Node::kGemm, {a, b, c});
        n->producer()->set_alpha(alpha)->set_beta(beta);
        return n->producer();
    };

    int64_t const out_size = 2 * 2;
    EXPECT_EQ(6 * out_size, CalculateFlops(*make_gemm(1.0, 0.0)));
    EXPECT_EQ(6 * out_size + out_size, CalculateFlops(*make_gemm(0.5, 0.0)));
    EXPECT_EQ((6 * out_size) + (out_size + 4), CalculateFlops(*make_gemm(1.0, 1.0)));
    EXPECT_EQ((6 * out_size) + (out_size + 4 + out_size), CalculateFlops(*make_gemm(1.0, 0.5)));
}

}  // namespace
}  // namespace chainer_compiler
