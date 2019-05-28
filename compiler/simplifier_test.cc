#include <gtest/gtest.h>

#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/simplifier.h>

namespace chainer_compiler {
namespace {

std::vector<Node::OpType> TestSimplify(const std::string& name, Node::OpType op_type, int num_inputs, int num_outputs) {
    Type type(Dtype::kFloat32, {});
    Graph graph("test");
    std::vector<Value*> inputs;
    for (int i = 0; i < num_inputs; ++i) {
        inputs.push_back(graph.AddInputValue(StrCat("input", i), type));
    }
    std::vector<Value*> outputs;
    for (int i = 0; i < num_outputs; ++i) {
        outputs.push_back(graph.AddOutputValue(StrCat("output", i), type));
    }

    {
        GraphBuilder gb(&graph, "test", outputs[0]);
        gb.MOp(op_type, inputs, outputs);
    }
    Simplify({name}, &graph, true  /* gen_backprop */);

    std::vector<Node::OpType> ops;
    for (Node* node : graph.GetLiveNodes()) {
        ops.push_back(node->op_type());
    }
    return ops;
}

TEST(SimplifyTest, Sum) {
    EXPECT_EQ(std::vector<Node::OpType>({Node::kIdentity}), TestSimplify("ReplaceSum", Node::kSum, 1, 1));
    EXPECT_EQ(std::vector<Node::OpType>({Node::kAdd}), TestSimplify("ReplaceSum", Node::kSum, 2, 1));
    EXPECT_EQ(std::vector<Node::OpType>({Node::kAdd, Node::kAdd}), TestSimplify("ReplaceSum", Node::kSum, 3, 1));
}

// TODO(hamaji): Write tests for other ops.

}  // namespace
}  // namespace chainer_compiler
