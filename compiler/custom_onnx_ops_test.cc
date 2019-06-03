#include <gtest/gtest.h>

#include <common/iterator.h>
#include <common/strutil.h>
#include <compiler/graph.h>
#include <compiler/graph_builder.h>
#include <compiler/custom_onnx_ops.h>

namespace chainer_compiler {
namespace {

void TestInference(
    Node::OpType op_type, const std::vector<Type>& input_types, Dtype expected_dtype, const std::vector<int64_t>& expected_dims) {
    Graph graph("test");
    std::vector<Value*> inputs;
    for (const auto& type : Enumerate(input_types)) {
        inputs.push_back(graph.AddInputValue(StrCat("input", type.index), type.value));
    }
    std::vector<Value*> outputs;
    outputs.push_back(graph.AddOutputValue(StrCat("output", 0), Type()));

    Node* node = nullptr;
    {
        GraphBuilder gb(&graph, "test", outputs[0]);
        node = gb.MOp(op_type, inputs, outputs);
    }

    ASSERT_EQ(1, node->outputs().size());
    const Type& output_type = node->output(0)->type();
    EXPECT_EQ(expected_dtype, output_type.dtype());
    EXPECT_EQ(expected_dims, output_type.dims());
}

std::vector<Type> Types(const std::vector<std::vector<int64_t>>& shapes) {
    std::vector<Type> types;
    for (const std::vector<int64_t>& shape : shapes) {
        types.push_back(Type(Dtype::kFloat32, shape));
    }
    return types;
}

TEST(ShapeInferenceTest, Linear) {
    RegisterCustomOnnxOperatorSetSchema();
    TestInference(Node::kChainerLinear, Types({{3, 2}, {4, 2}}), Dtype::kFloat32, {3, 4});
}

}  // namespace
}  // namespace chainer_compiler
