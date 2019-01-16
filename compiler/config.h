#pragma once

#include <memory>
#include <string>

#include <compiler/node.h>

namespace chainer_compiler {

class CompilerConfig {
public:
    virtual ~CompilerConfig() = default;

    virtual bool HasOp(Node::OpType op) const = 0;

    virtual std::string name() const = 0;

protected:
    CompilerConfig() = default;
};

std::unique_ptr<CompilerConfig> GetCompilerConfig(const std::string& backend_name);

}  // namespace chainer_compiler
