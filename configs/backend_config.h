#pragma once

#include <memory>
#include <set>
#include <string>

namespace chainer_compiler {

class BackendConfig {
public:
    static std::unique_ptr<BackendConfig> FromName(const std::string& name);
    static std::unique_ptr<BackendConfig> FromJSON(const std::string& json_str);

    virtual ~BackendConfig() = default;

    virtual const std::string& name() const = 0;

    virtual const std::set<std::string>& GetSimplifyPreproc() const = 0;
    virtual const std::set<std::string>& GetSimplify() const = 0;
    virtual bool HasOp(const std::string& name) const = 0;
    virtual const std::set<std::string>& GetMerge() const = 0;
};

}  // namespace chainer_compiler
