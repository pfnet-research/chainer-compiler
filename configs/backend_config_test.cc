#include <gtest/gtest.h>

#include <configs/backend_config.h>

namespace chainer_compiler {
namespace {

TEST(BackendConfigTest, ChxVM) {
    std::unique_ptr<BackendConfig> config = BackendConfig::FromName("chxvm");
    EXPECT_EQ("chxvm", config->name());
    EXPECT_EQ(0, config->GetSimplifyPreproc().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplifyPreproc().count("NOT FOUND"));
    EXPECT_EQ(0, config->GetSimplifyPreproc().count("ReplaceChainerLinear"));
    EXPECT_EQ(1, config->GetSimplifyPreproc().count("ReplaceChainerSelectItem"));
    EXPECT_EQ(1, config->GetSimplify().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplify().count("NOT FOUND"));
    EXPECT_EQ(0, config->GetSimplify().count("ReplaceChainerLinear"));
    EXPECT_EQ(1, config->GetSimplify().count("ReplaceChainerSelectItem"));
}

TEST(BackendConfigTest, ChxVMTest) {
    std::unique_ptr<BackendConfig> config = BackendConfig::FromName("chxvm_test");
    EXPECT_EQ("chxvm_test", config->name());
    EXPECT_EQ(0, config->GetSimplifyPreproc().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplifyPreproc().count("NOT FOUND"));
    EXPECT_EQ(1, config->GetSimplifyPreproc().count("ReplaceChainerLinear"));
    EXPECT_EQ(0, config->GetSimplifyPreproc().count("ReplaceChainerSelectItem"));
    EXPECT_EQ(1, config->GetSimplify().count("ReplaceLess"));
    EXPECT_EQ(0, config->GetSimplify().count("NOT FOUND"));
    EXPECT_EQ(1, config->GetSimplify().count("ReplaceChainerLinear"));
    EXPECT_EQ(0, config->GetSimplify().count("ReplaceChainerSelectItem"));
}

}  // namespace
}  // namespace chainer_compiler
