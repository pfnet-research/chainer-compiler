#pragma once

#include <nonstd/optional.hpp>

#include <chainerx/array.h>

namespace oniku {
namespace runtime {

typedef std::vector<nonstd::optional<chainerx::Array>> XCVMSequence;

class XCVMOpaque {
public:
    virtual ~XCVMOpaque() = default;

    virtual std::string ToString() const { return "???"; }
    virtual std::string DebugString() const { return "???"; }

protected:
    XCVMOpaque() = default;
};

class XCVMVar {
public:
    enum class Kind {
        kArray,
        kSequence,
        kOpaque,
        kNull,
    };

    explicit XCVMVar(Kind kind);
    explicit XCVMVar(chainerx::Array array);
    // Takes the ownership of `opaque`.
    explicit XCVMVar(XCVMOpaque* opaque);
    explicit XCVMVar(const XCVMVar&) = default;

    const chainerx::Array& GetArray();
    XCVMSequence* GetSequence();
    XCVMOpaque* GetOpaque();

    Kind kind() const {
        return kind_;
    }

    int64_t GetTotalSize() const;

    char Sigil() const;

    std::string ToString() const;
    std::string DebugString() const;

private:
    Kind kind_;
    chainerx::Array array_;
    XCVMSequence sequence_;
    std::shared_ptr<XCVMOpaque> opaque_;
};

std::vector<chainerx::Array> NonOptional(const XCVMSequence& v);

}  // namespace runtime
}  // namespace oniku
