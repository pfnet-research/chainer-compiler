#pragma once

#include <nonstd/optional.hpp>

#include <chainerx/array.h>

namespace oniku {
namespace runtime {

typedef std::vector<nonstd::optional<chainerx::Array>> XCVMSequence;

class XCVMVar {
public:
    enum class Kind {
        kArray,
        kSequence,
    };

    explicit XCVMVar(Kind kind);
    explicit XCVMVar(chainerx::Array array);
    explicit XCVMVar(const XCVMVar&) = default;

    const chainerx::Array& GetArray();
    XCVMSequence* GetSequence();

    Kind kind() const {
        return kind_;
    }

    int64_t GetTotalSize() const;

    char Sigil() const;

    std::string ToString() const;
    std::string DebugString() const;

private:
    Kind kind_;
    nonstd::optional<chainerx::Array> array_;
    XCVMSequence sequence_;
};

}  // namespace runtime
}  // namespace oniku
