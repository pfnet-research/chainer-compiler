#pragma once

#include <nonstd/optional.hpp>

#include <chainerx/array.h>

namespace chainer_compiler {
namespace runtime {

class ChxVMVar;

typedef std::vector<ChxVMVar> ChxVMSequence;

class ChxVMOpaque {
public:
    virtual ~ChxVMOpaque() = default;

    virtual std::string ToString() const {
        return "???";
    }
    virtual std::string DebugString() const {
        return "???";
    }

    const std::vector<chainerx::Array>& GetArrays() const;

    void SetRetainedArrays(const std::vector<chainerx::Array>& retained_arrays);

protected:
    ChxVMOpaque() = default;

    std::unique_ptr<std::vector<chainerx::Array>> retained_arrays_;
};

class ChxVMVar {
public:
    enum class Kind {
        kArray,
        kSequence,
        kOpaque,
        kNull,
    };

    ChxVMVar();
    explicit ChxVMVar(Kind kind);
    explicit ChxVMVar(chainerx::Array array);
    // Takes the ownership of `opaque`.
    explicit ChxVMVar(ChxVMOpaque* opaque);
    explicit ChxVMVar(const ChxVMVar&) = default;

    const chainerx::Array& GetArray() const;
    ChxVMSequence* GetSequence() const;
    ChxVMOpaque* GetOpaque() const;

    Kind kind() const {
        return kind_;
    }
    bool IsNull() const {
        return kind_ == Kind::kNull;
    }

    int64_t GetNBytes() const;

    std::vector<chainerx::Array> GetArrays() const;

    char Sigil() const;

    std::string ToString() const;
    std::string DebugString() const;

private:
    Kind kind_;
    chainerx::Array array_;
    std::shared_ptr<ChxVMSequence> sequence_;
    std::shared_ptr<ChxVMOpaque> opaque_;
};

std::vector<chainerx::Array> NonOptional(const ChxVMSequence& seq);

std::ostream& operator<<(std::ostream& os, const ChxVMVar::Kind& kind);

}  // namespace runtime
}  // namespace chainer_compiler
