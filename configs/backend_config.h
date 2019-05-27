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

    virtual const std::set<std::string>& GetSimplifyAlways() const = 0;
    virtual const std::set<std::string>& GetSimplifyFull() const = 0;
};

}  // namespace chainer_compiler
