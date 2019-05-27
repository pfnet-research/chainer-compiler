#include <configs/backend_config.h>

#include <iostream>

#include <common/log.h>
#include <configs/json_repository.h>

namespace chainer_compiler {

class BackendConfigImpl : public BackendConfig {
public:
    explicit BackendConfigImpl(const json& config) {
        CHECK(config.is_object()) << config;
        for (const auto& el : config.items()) {
            if (el.key() == "simplify") {
                ParseSimplify(el.value());
            } else {
                std::cerr << "WARNING: Unknown backend config: " << el.key() << std::endl;
            }
        }
    }

    ~BackendConfigImpl() override = default;

    const std::set<std::string>& GetSimplifiers() const override {
        return simplifilers_;
    }

private:
    void ParseSimplify(const json& simplify) {
        CHECK(simplify.is_object()) << "simplify must be an object: " << simplify;
        for (const auto& el : simplify.items()) {
            CHECK(el.value().is_boolean()) << "simplify values must be bool: " << simplify;
            if (el.value() == false) {
                continue;
            }
            CHECK(simplifilers_.emplace(el.key()).second) << "Duplicate key: " << el.key();
        }
    }

    std::set<std::string> simplifilers_;
};

std::unique_ptr<BackendConfig> BackendConfig::FromName(const std::string& name) {
    json j = LoadJSONFromName(name);
    return std::make_unique<BackendConfigImpl>(j);
}

std::unique_ptr<BackendConfig> BackendConfig::FromJSON(const std::string& json_str) {
    json j = LoadJSONFromString(json_str);
    return std::make_unique<BackendConfigImpl>(j);
}

}  // namespace chainer_compiler
