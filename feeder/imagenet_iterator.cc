#include "imagenet_iterator.h"

#include <cstring>
#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include <xchainer/routines/creation.h>

#include <common/log.h>
#include <common/strutil.h>

namespace {

xchainer::Array MakeArray(xchainer::Dtype dtype, xchainer::Shape shape, const void* src) {
    int64_t size = xchainer::GetItemSize(dtype) * shape.GetTotalSize();
    std::shared_ptr<void> data(new char[size], std::default_delete<char[]>());
    std::memcpy(data.get(), src, size);
    xchainer::Array array(xchainer::FromContiguousHostData(shape, dtype, data));
    return array;
}

}  // namespace

ImageNetIterator::ImageNetIterator(const std::string& labeled_image_dataset, int buf_size, int batch_size, const std::vector<float>& mean, int height, int width)
    : DataIterator(buf_size),
      batch_size_(batch_size),
      mean_(mean),
      height_(height),
      width_(width) {
    CHECK_EQ(3 * height * width, mean_.size());
    std::ifstream ifs(labeled_image_dataset);
    while (ifs) {
        std::string filename;
        int label;
        ifs >> filename >> label;
        dataset_.emplace_back(filename, label);
    }
    // std::cerr << dataset_.size() << " examples" << std::endl;
}

std::vector<xchainer::Array> ImageNetIterator::GetNextImpl() {
    std::vector<std::pair<std::string, int>> batch;
    while (batch_size_ > batch.size()) {
        if (iter_ == dataset_.size()) {
            std::cerr << batch.size() << std::endl;
            break;
        }
        batch.push_back(dataset_[iter_++]);
    }
    if (batch.empty())
        return {};

    std::vector<float> image_data(batch.size() * 3 * height_ * width_);
    std::vector<int> label_data(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        label_data[i] = batch[i].second;
        cv::Mat image = cv::imread(batch[i].first);
        int bi = i * 3 * height_ * width_;
        CHECK_GE(image.rows, height_);
        CHECK_GE(image.cols, width_);
        int by = (image.rows - height_) / 2;
        int bx = (image.cols - width_) / 2;
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                int ii = (y * width_ + x) * 3;
                for (int k = 0; k < 3; ++k) {
                    image_data[bi + ii + k] = image.at<cv::Vec3b>(by + y, bx + x)(2 - k);
                    image_data[bi + ii + k] -= mean_[ii + k];
                    image_data[bi + ii + k] *= 1.0 / 255.0;
                }
            }
        }
    }

    std::vector<xchainer::Array> arrays;
    int bs = static_cast<int>(batch.size());
    arrays.push_back(MakeArray(xchainer::Dtype::kFloat32, {bs, 3, height_, width_}, image_data.data()));
    arrays.push_back(MakeArray(xchainer::Dtype::kInt32, {bs}, label_data.data()));
    return arrays;
}

std::string ImageNetIterator::GetStatus() const {
    return oniku::StrCat(iter_, "/", dataset_.size());
}

std::vector<float> LoadMean(const std::string& filename, int height, int width) {
    const int ORIG_HEIGHT = 256;
    const int ORIG_WIDTH = 256;
    CHECK_GE(ORIG_HEIGHT, height);
    CHECK_GE(ORIG_WIDTH, width);
    std::ifstream ifs(filename);
    CHECK(ifs) << "Failed to open: " << filename;
    int num_elements = 3 * ORIG_HEIGHT * ORIG_WIDTH;
    std::vector<float> mean(num_elements);
    ifs.read(reinterpret_cast<char*>(&mean[0]), sizeof(float) * num_elements);
    CHECK_EQ(sizeof(float) * num_elements, ifs.gcount()) << "Invalid mean file: " << filename;

    std::vector<float> cropped(3 * height * width);
    int by = (ORIG_HEIGHT - height) / 2;
    int bx = (ORIG_WIDTH - width) / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int k = 0; k < 3; ++k) {
                cropped[(y * width + x) * 3 + k] = mean[((by + y) * width + (bx + x)) * 3 + k];
            }
        }
    }
    return cropped;
}
