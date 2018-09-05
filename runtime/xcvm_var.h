#pragma once

#include <nonstd/optional.hpp>

#include <xchainer/array.h>

namespace oniku {
namespace runtime {

class XCVMVar {
public:
    enum class Kind {
        kArray,
        kSequence,
    };

    explicit XCVMVar(Kind kind);
    explicit XCVMVar(xchainer::Array array);

    xchainer::Array GetArray();
    std::vector<xchainer::Array>* GetSequence();

    std::string ToString() const;
    std::string DebugString() const;

private:
    Kind kind_;
    nonstd::optional<xchainer::Array> array_;
    std::vector<xchainer::Array> sequence_;
};

}  // namespace runtime
}  // namespace oniku
