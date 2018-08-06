#include <string>

#include <gtest/gtest.h>

#include <common/log.h>
#include <common/protoutil.h>
#include <compiler/tensor.h>

namespace oniku {
namespace {

TEST(TensorTest, LoadMNISTOutput) {
    std::string path = "../data/mnist/test_data_set_0/output_0.pb";
    onnx::TensorProto xtensor(LoadLargeProto<onnx::TensorProto>(path));
    Tensor tensor(xtensor);
    ASSERT_EQ(2, tensor.dims().size());
    EXPECT_EQ(1, tensor.dims()[0]);
    EXPECT_EQ(10, tensor.dims()[1]);
    EXPECT_EQ(Tensor::Dtype::kFloat32, tensor.dtype());
}

}  // namespace
}  // namespace oniku
