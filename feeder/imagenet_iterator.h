#pragma once

#include <string>
#include <utility>
#include <vector>

#include <xchainer/array.h>

#include <feeder/data_iterator.h>

class ImageNetIterator : public DataIterator {
public:
    explicit ImageNetIterator(const std::string& labeled_image_dataset, int buf_size, int batch_size, const std::vector<float>& mean, int height, int width);

    std::vector<xchainer::Array> GetNextImpl() override;

    std::string GetStatus() const;

private:
    std::vector<std::pair<std::string, int>> dataset_;
    size_t iter_ = 0;
    int batch_size_;
    std::vector<float> mean_;
    int height_;
    int width_;
};

std::vector<float> LoadMean(const std::string& filename, int height, int width);
