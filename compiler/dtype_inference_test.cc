#include <gtest/gtest.h>

#include <common/log.h>
#include <compiler/dtype.h>
#include <compiler/dtype_inference.h>

namespace oniku {
namespace {

TEST(ShapeInferenceTest, CoerceDtype) {
    EXPECT_EQ(Dtype::kInt32, CoerceDtype(Dtype::kInt32, Dtype::kInt32));
    EXPECT_EQ(Dtype::kUnknown, CoerceDtype(Dtype::kFloat32, Dtype::kUnknown));
    EXPECT_EQ(Dtype::kFloat32, CoerceDtype(Dtype::kFloat32, Dtype::kInt64));
    EXPECT_EQ(Dtype::kFloat32, CoerceDtype(Dtype::kInt64, Dtype::kFloat32));
    EXPECT_EQ(Dtype::kFloat64, CoerceDtype(Dtype::kFloat64, Dtype::kFloat32));
    EXPECT_EQ(Dtype::kFloat64, CoerceDtype(Dtype::kFloat32, Dtype::kFloat64));
    EXPECT_EQ(Dtype::kInt64, CoerceDtype(Dtype::kInt64, Dtype::kInt16));
    EXPECT_EQ(Dtype::kInt16, CoerceDtype(Dtype::kInt8, Dtype::kInt16));
    EXPECT_EQ(Dtype::kInt8, CoerceDtype(Dtype::kInt8, Dtype::kBool));
    EXPECT_EQ(Dtype::kUInt8, CoerceDtype(Dtype::kBool, Dtype::kUInt8));
    EXPECT_EQ(Dtype::kInt16, CoerceDtype(Dtype::kInt8, Dtype::kUInt8));
    EXPECT_EQ(Dtype::kInt16, CoerceDtype(Dtype::kUInt8, Dtype::kInt8));
}

TEST(ShapeInferenceTest, CoerceDtypeAll) {
    std::vector<Dtype> all_types = {
            Dtype::kUnknown,
            Dtype::kBool,
            Dtype::kInt8,
            Dtype::kInt16,
            Dtype::kInt32,
            Dtype::kInt64,
            Dtype::kUInt8,
            Dtype::kFloat32,
            Dtype::kFloat64,
    };

    // All combinations should not crash and reflective.
    for (Dtype dtype0 : all_types) {
        for (Dtype dtype1 : all_types) {
            Dtype r0 = CoerceDtype(dtype0, dtype1);
            Dtype r1 = CoerceDtype(dtype1, dtype0);
            EXPECT_EQ(r0, r1);
        }
    }
}

}  // namespace
}  // namespace oniku
