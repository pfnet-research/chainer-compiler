#include <common/log.h>

#include <cstdlib>
#include <iostream>

namespace oniku {

FailMessageStream::FailMessageStream(const std::string msg, const char* func, const char* file, int line, bool is_check)
    : msg_(msg), func_(func), file_(file), line_(line), is_check_(is_check) {
}

FailMessageStream::~FailMessageStream() {
    if (is_check_) {
        std::cerr << msg_ << " in " << func_ << " at " << file_ << ":" << line_ << ": " << oss_.str() << std::endl;
        std::abort();
    } else {
        std::cerr << oss_.str() << std::endl;
        std::exit(1);
    }
}

}  // namespace oniku
