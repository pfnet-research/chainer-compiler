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

#define CHECK_EQ(a, b) \
    while ((a) != (b)) \
    oniku::FailMessageStream("Check `" #a "' == `" #b "' failed!", __func__, __FILE__, __LINE__) << "(" << a << " vs " << b << ")"

#define CHECK_NE(a, b) \
    while ((a) == (b)) \
    oniku::FailMessageStream("Check `" #a "' != `" #b "' failed!", __func__, __FILE__, __LINE__) << "(" << a << " vs " << b << ")"

#ifdef NDEBUG
#define DCHECK(cond)
#define DCHECK_EQ(a, b)
#define DCHECK_NE(a, b)
#else
#define DCHECK(cond) CHECK(cond)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#endif

}  // namespace oniku
