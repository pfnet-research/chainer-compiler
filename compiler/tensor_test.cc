#include <string>

#include <gtest/gtest.h>

#include <chainerx/testing/context_session.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/dtype.h>
#include <compiler/tensor.h>

namespace chainer_compiler {
namespace {

TEST(TensorTest, LoadMNISTOutput) {
    chainerx::testing::ContextSession sess;
    std::string path = "data/mnist/test_data_set_0/output_0.pb";
    onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(path));
    Tensor tensor(xtensor);
    ASSERT_EQ(2, tensor.dims().size());
    EXPECT_EQ(1, tensor.dims()[0]);
    EXPECT_EQ(10, tensor.dims()[1]);
    EXPECT_EQ(Dtype::kFloat32, tensor.dtype());
}

TEST(TensorTest, Constructor) {
    chainerx::testing::ContextSession sess;
    {
        Tensor tensor("foo", Dtype::kFloat32, {2}, {2.0f, 3.0f});
        EXPECT_EQ(2.0, tensor.Get<float>(0));
        EXPECT_EQ(3.0, tensor.Get<float>(1));
    }
    {
        Tensor tensor("foo", Dtype::kFloat32, {2}, {2, 3});
        EXPECT_EQ(2.0, tensor.Get<float>(0));
        EXPECT_EQ(3.0, tensor.Get<float>(1));
    }
    {
        Tensor tensor("foo", Dtype::kBool, {2}, {2.0, 0.0});
        EXPECT_TRUE(tensor.Get<bool>(0));
        EXPECT_FALSE(tensor.Get<bool>(1));
    }
    {
        Tensor tensor("foo", Dtype::kFloat64, {2}, {2, 3});
        EXPECT_EQ(2.0, tensor.Get<double>(0));
        EXPECT_EQ(3.0, tensor.Get<double>(1));
    }
}

}  // namespace
}  // namespace chainer_compiler
