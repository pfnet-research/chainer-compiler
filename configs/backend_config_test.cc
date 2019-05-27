#include <gtest/gtest.h>

#include <configs/backend_config.h>

namespace chainer_compiler {
namespace {

TEST(BackendConfigTest, ChxVM) {
    std::unique_ptr<BackendConfig> config = BackendConfig::FromName("chxvm");
    EXPECT_EQ(0, config->GetSimplifyAlways().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplifyAlways().count("NOT FOUND"));
    EXPECT_EQ(0, config->GetSimplifyAlways().count("ReplaceChainerLinear"));
    EXPECT_EQ(1, config->GetSimplifyAlways().count("ReplaceChainerSelectItem"));
    EXPECT_EQ(1, config->GetSimplifyFull().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplifyFull().count("NOT FOUND"));
    EXPECT_EQ(0, config->GetSimplifyFull().count("ReplaceChainerLinear"));
    EXPECT_EQ(1, config->GetSimplifyFull().count("ReplaceChainerSelectItem"));
}

TEST(BackendConfigTest, ChxVMTest) {
    std::unique_ptr<BackendConfig> config = BackendConfig::FromName("chxvm_test");
    EXPECT_EQ(0, config->GetSimplifyAlways().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplifyAlways().count("NOT FOUND"));
    EXPECT_EQ(1, config->GetSimplifyAlways().count("ReplaceChainerLinear"));
    EXPECT_EQ(0, config->GetSimplifyAlways().count("ReplaceChainerSelectItem"));
    EXPECT_EQ(1, config->GetSimplifyFull().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplifyFull().count("NOT FOUND"));
    EXPECT_EQ(1, config->GetSimplifyFull().count("ReplaceChainerLinear"));
    EXPECT_EQ(0, config->GetSimplifyFull().count("ReplaceChainerSelectItem"));
}

}  // namespace
}  // namespace chainer_compiler
