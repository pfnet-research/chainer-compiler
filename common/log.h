#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace oniku {

class FailMessageStream {
public:
    FailMessageStream(const std::string msg, const char* func, const char* file, int line, bool is_check = true)
        : msg_(msg), func_(func), file_(file), line_(line), is_check_(is_check) {
    }

    ~FailMessageStream() {
        if (is_check_) {
            std::cerr << msg_ << " in " << func_ << " at " << file_ << ":" << line_ << ": " << oss_.str() << std::endl;
            std::abort();
        } else {
            std::cerr << oss_.str() << std::endl;
            std::exit(1);
        }
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
    const bool is_check_;
};

#define QFAIL() oniku::FailMessageStream("", __func__, __FILE__, __LINE__, false)

#define CHECK(cond) \
    while (!(cond)) oniku::FailMessageStream("Check `" #cond "' failed!", __func__, __FILE__, __LINE__)

#define CHECK_CMP(a, b, op) \
    while (!((a)op(b)))     \
    oniku::FailMessageStream("Check `" #a "' " #op " `" #b "' failed!", __func__, __FILE__, __LINE__) << "(" << a << " vs " << b << ")"

#define CHECK_EQ(a, b) CHECK_CMP(a, b, ==)
#define CHECK_NE(a, b) CHECK_CMP(a, b, !=)
#define CHECK_LT(a, b) CHECK_CMP(a, b, <)
#define CHECK_LE(a, b) CHECK_CMP(a, b, <=)
#define CHECK_GT(a, b) CHECK_CMP(a, b, >)
#define CHECK_GE(a, b) CHECK_CMP(a, b, >=)

#ifdef NDEBUG
#define DCHECK(cond)
#define DCHECK_EQ(a, b)
#define DCHECK_NE(a, b)
#define DCHECK_LT(a, b)
#define DCHECK_LE(a, b)
#define DCHECK_GT(a, b)
#define DCHECK_GE(a, b)
#else
#define DCHECK(cond) CHECK(cond)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#endif

#define WARN_ONCE(msg)                                                        \
    do {                                                                      \
        static bool logged_##__LINE__ = false;                                \
        if (!logged_##__LINE__) std::cerr << "WARNING: " << msg << std::endl; \
        logged_##__LINE__ = true;                                             \
    } while (0)

}  // namespace oniku
