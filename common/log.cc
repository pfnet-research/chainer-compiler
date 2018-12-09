#include <common/log.h>

#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>

namespace oniku {
namespace {

void MaybeWait() {
    if (!std::getenv("ONIKU_WAIT_ON_CRASH")) return;
    std::cout << "Waiting before crash. You can attach gdb by\n"
              << "$ gdb -p " << getpid() << std::endl;
    std::string line;
    std::getline(std::cin, line);
}

}  // namespace

FailMessageStream::FailMessageStream(const std::string msg, const char* func, const char* file, int line, bool is_check)
    : msg_(msg), func_(func), file_(file), line_(line), is_check_(is_check) {
}

FailMessageStream::~FailMessageStream() {
    if (is_check_) {
        std::cerr << msg_ << " in " << func_ << " at " << file_ << ":" << line_ << ": " << oss_.str() << std::endl;
        MaybeWait();
        std::abort();
    } else {
        std::cerr << oss_.str() << std::endl;
        MaybeWait();
        std::exit(1);
    }
}

}  // namespace oniku
