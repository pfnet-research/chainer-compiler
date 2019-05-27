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
            if (el.key() == "simplify_always") {
                ParseSimplify(el.value(), &simplify_always_);
            } else if (el.key() == "simplify_full") {
                ParseSimplify(el.value(), &simplify_full_);
            } else {
                std::cerr << "WARNING: Unknown backend config: " << el.key() << std::endl;
            }
        }

        for (const std::string& n : simplify_always_) {
            simplify_full_.emplace(n);
        }
    }

    ~BackendConfigImpl() override = default;

    const std::set<std::string>& GetSimplifyAlways() const override {
        return simplify_always_;
    }

    const std::set<std::string>& GetSimplifyFull() const override {
        return simplify_full_;
    }

private:
    void ParseSimplify(const json& simplify, std::set<std::string>* names) {
        CHECK(simplify.is_object()) << "simplify must be an object: " << simplify;
        for (const auto& el : simplify.items()) {
            CHECK(el.value().is_boolean()) << "simplify values must be bool: " << simplify;
            if (el.value() == false) {
                continue;
            }
            CHECK(names->emplace(el.key()).second) << "Duplicate key: " << el.key();
        }
    }

    std::set<std::string> simplify_always_;
    std::set<std::string> simplify_full_;
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
