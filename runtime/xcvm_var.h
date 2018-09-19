#pragma once

#include <nonstd/optional.hpp>

#include <chainerx/array.h>

namespace oniku {
namespace runtime {

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
    std::vector<chainerx::Array>* GetSequence();

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
    std::vector<chainerx::Array> sequence_;
};

}  // namespace runtime
}  // namespace oniku
