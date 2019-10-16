#include <gtest/gtest.h>

#include <compiler/flags.h>
#include <compiler/fusion.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>

namespace chainer_compiler {
namespace {

TEST(FusionTest, Basic) {
    // TODO(hamaji): Introduce something like CompilerContext.
    g_fuse_operations = true;
    Type type(Dtype::kFloat32, {});
    Graph graph({}, "test");
    Value* input = graph.AddInputValue("input", type);
    Value* output = graph.AddOutputValue("output", type);
    GraphBuilder gb(&graph, "test", output);
    Value* tmp = gb.Op(Node::kTanh, {input});
    gb.Op(Node::kSigmoid, {tmp}, {output});

    FuseOperations(&graph);
    ASSERT_EQ(1, graph.nodes().size());
    const Node& node = *graph.nodes()[0];
    ASSERT_EQ(Node::kChainerFusionGroup, node.op_type());
    ASSERT_TRUE(node.subgraph());
    EXPECT_EQ(2, node.subgraph()->nodes().size());
    graph.CheckSanity("fused");
    g_fuse_operations = false;
}

}  // namespace
}  // namespace chainer_compiler
