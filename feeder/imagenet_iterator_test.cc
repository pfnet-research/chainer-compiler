#include <fstream>

#include <gtest/gtest.h>

#include <xchainer/context.h>
#include <xchainer/routines/manipulation.h>

#include <common/log.h>
#include <feeder/imagenet_iterator.h>

namespace {

bool file_exists(const std::string& filename) {
    std::ifstream ifs(filename);
    return ifs.good();
}

TEST(TestImageNetIterator, Basic) {
    // Prepare data by:
    //
    // import numpy as np
    // import struct
    // m = np.load('data/imagenet/mean.npy')
    //   with open('data/imagenet/mean.bin', 'wb') as f:
    //   f.write(struct.pack('%df' % m.size, *tuple(m.flat)))
    if (!file_exists("data/imagenet/test.txt") ||
        !file_exists("data/imagenet/mean.bin")) {
        WARN_ONCE("Test skipped");
        return;
    }

    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);

    std::vector<float> mean(LoadMean("data/imagenet/mean.bin", 192, 192));
    ASSERT_EQ(192 * 192 * 3, mean.size());
    EXPECT_LT(0, mean[0]);
    EXPECT_GT(256, mean[0]);

    ImageNetIterator iter("data/imagenet/test.txt", 3, 5, mean, 192, 192);
    iter.Start();
    std::vector<xchainer::Array> a(iter.GetNext());
    ASSERT_EQ(2, a.size());
    EXPECT_EQ(xchainer::Shape({5, 3, 192, 192}), a[0].shape());
    EXPECT_EQ(xchainer::Shape({5}), a[1].shape());
    EXPECT_EQ(0, int(xchainer::AsScalar(a[1].At({2}))));
    iter.Terminate();
}

}  // namespace
