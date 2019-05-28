#include <gtest/gtest.h>

#include <common/iterator.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/simplifier.h>

namespace chainer_compiler {
namespace {

std::vector<Node::OpType> TestSimplify(const std::string& name, Node::OpType op_type, const std::vector<Type>& input_types, const std::vector<Type>& output_types) {
    Graph graph("test");
    std::vector<Value*> inputs;
    for (const auto& type : Enumerate(input_types)) {
        inputs.push_back(graph.AddInputValue(StrCat("input", type.index), type.value));
    }
    std::vector<Value*> outputs;
    for (const auto& type : Enumerate(output_types)) {
        outputs.push_back(graph.AddOutputValue(StrCat("output", type.index), type.value));
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

std::vector<Node::OpType> TestSimplify(const std::string& name, Node::OpType op_type, int num_inputs, int num_outputs) {
    std::vector<Type> input_types;
    for (int i = 0; i < num_inputs; ++i) {
        input_types.push_back(Type(Dtype::kFloat32, {}));
    }
    std::vector<Type> output_types;
    for (int i = 0; i < num_outputs; ++i) {
        output_types.push_back(Type(Dtype::kFloat32, {}));
    }
    return TestSimplify(name, op_type, input_types, output_types);
}

std::vector<Type> MakeTypes(std::vector<std::vector<int64_t>> shapes) {
    std::vector<Type> types;
    for (const std::vector<int64_t>& shape : shapes) {
        types.push_back(Type(Dtype::kFloat32, shape));
    }
    return types;
}

TEST(SimplifyTest, Sum) {
    EXPECT_EQ(std::vector<Node::OpType>({Node::kIdentity}), TestSimplify("ReplaceSum", Node::kSum, 1, 1));
    EXPECT_EQ(std::vector<Node::OpType>({Node::kAdd}), TestSimplify("ReplaceSum", Node::kSum, 2, 1));
    EXPECT_EQ(std::vector<Node::OpType>({Node::kAdd, Node::kAdd}), TestSimplify("ReplaceSum", Node::kSum, 3, 1));
}

TEST(SimplifyTest, Split) {
    EXPECT_EQ(std::vector<Node::OpType>({Node::kSlice, Node::kSlice, Node::kSlice}), TestSimplify("ReplaceSplit", Node::kSplit, MakeTypes({{3}}), MakeTypes({{1}, {1}, {1}})));
}

// TODO(hamaji): Write tests for other ops.

}  // namespace
}  // namespace chainer_compiler
