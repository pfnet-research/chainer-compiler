#pragma once

#include <memory>
#include <string>

#include <chainerx/array.h>
#include <chainerx/backprop_scope.h>

#include <runtime/chxvm_var.h>

namespace chainer_compiler {
namespace runtime {

class BackwardContext : public ChxVMOpaque {
public:
    BackwardContext(const std::string& name, const std::vector<chainerx::Array>& xs);
    virtual ~BackwardContext() = default;

    chainerx::BackpropId backprop_id() const {
        return backprop_->backprop_id();
    }

    void SetOutput(const std::vector<chainerx::Array>& ys);

    std::vector<chainerx::Array> Backward(const std::vector<chainerx::Array>& gys) const;

private:
    std::unique_ptr<chainerx::BackpropScope> backprop_;
    const std::vector<chainerx::Array> xs_;
    std::vector<chainerx::Array> ys_;
};

}  // namespace runtime
}  // namespace chainer_compiler
