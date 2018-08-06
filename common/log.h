#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace oniku {

class FailMessageStream {
public:
    FailMessageStream(const std::string msg, const char* func, const char* file, int line)
        : msg_(msg), func_(func), file_(file), line_(line) {}

    ~FailMessageStream() {
        std::cerr << msg_ << " in " << func_ << " at " << file_ << ":" << line_ << ": " << oss_.str() << std::endl;
        std::abort();
    }

    template <class T>
    FailMessageStream& operator<<(const T& v) {
        oss_ << v;
        return *this;
    }

private:
    std::ostringstream oss_;
    const std::string msg_;
    const char* func_;
    const char* file_;
    const int line_;
};

#define CHECK(cond) \
    while (!(cond)) oniku::FailMessageStream("Check `" #cond "' failed!", __func__, __FILE__, __LINE__)

#ifndef NDEBUG
#define DCHECK(cond)
#endif

}  // namespace oniku
