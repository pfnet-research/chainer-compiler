#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <chainerx/array.h>

class DataIterator {
public:
    virtual ~DataIterator();

    std::vector<chainerx::Array> GetNext();

    virtual std::vector<chainerx::Array> GetNextImpl() = 0;

    void Start();
    void Terminate();

protected:
    explicit DataIterator(int buf_size);

private:
    void Loop();

    std::unique_ptr<std::thread> thread_;
    std::mutex mu_;
    std::condition_variable cond_;
    std::queue<std::vector<chainerx::Array>> buf_;
    const int buf_size_;
    bool should_finish_ = false;
    bool is_iteration_finished_ = false;
};
