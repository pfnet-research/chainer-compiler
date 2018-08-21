#include <cstring>
#include <memory>

#include <gtest/gtest.h>

#include <xchainer/array.h>
#include <xchainer/context.h>
#include <xchainer/routines/creation.h>
#include <xchainer/routines/manipulation.h>

#include <feeder/data_iterator.h>

namespace {

class MyDataIterator : public DataIterator {
public:
    explicit MyDataIterator(int end=999)
        : DataIterator(3), end_(end) {
    }

    std::vector<xchainer::Array> GetNextImpl() override {
        if (counter_ == end_)
            return {};
        std::shared_ptr<void> data(new char[sizeof(counter_)], std::default_delete<char[]>());
        std::memcpy(data.get(), &counter_, sizeof(counter_));
        xchainer::Array array = xchainer::FromContiguousHostData({}, xchainer::Dtype::kInt32, data);
        counter_++;
        return {array};
    }

private:
    int counter_ = 42;
    int end_;
};

TEST(TestDataIterator, Basic) {
    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);
    MyDataIterator iter;
    iter.Start();
    EXPECT_EQ(42, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    EXPECT_EQ(43, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    EXPECT_EQ(44, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    EXPECT_EQ(45, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    EXPECT_EQ(46, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    iter.Terminate();
}

TEST(TestDataIterator, Finish) {
    xchainer::Context ctx;
    xchainer::SetGlobalDefaultContext(&ctx);
    MyDataIterator iter(45);
    iter.Start();
    EXPECT_EQ(42, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    EXPECT_EQ(43, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    EXPECT_EQ(44, int64_t(xchainer::AsScalar(iter.GetNext()[0])));
    EXPECT_TRUE(iter.GetNext().empty());
    iter.Terminate();
}

}  // namespace
