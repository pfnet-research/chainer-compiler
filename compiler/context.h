#pragma once

#include <memory>
#include <string>

#include <compiler/node.h>

namespace oniku {

class CompilerContext {
public:
    virtual ~CompilerContext() = default;

    virtual bool HasOp(Node::OpType op) const = 0;

    virtual std::string name() const = 0;

protected:
    CompilerContext() = default;
};

std::unique_ptr<CompilerContext> GetCompilerContext(const std::string& backend_name);

}  // namespace oniku
