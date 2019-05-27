#include <configs/backend_config.h>

#include <iostream>

#include <common/log.h>
#include <configs/json_repository.h>

namespace chainer_compiler {

class BackendConfigImpl : public BackendConfig {
public:
    explicit BackendConfigImpl(const std::string& name, const json& config)
        : name_(name) {
        CHECK(config.is_object()) << config;
        for (const auto& el : config.items()) {
            if (el.key() == "simplify_preproc") {
                ParseSimplify(el.value(), &simplify_preproc_);
            } else if (el.key() == "simplify") {
                ParseSimplify(el.value(), &simplify_);
            } else if (el.key() == "supported_ops") {
                ParseSupportedOps(el.value(), &supported_ops_);
            } else {
                std::cerr << "WARNING: Unknown backend config: " << el.key() << std::endl;
            }
        }

        for (const std::string& n : simplify_preproc_) {
            simplify_.emplace(n);
        }
    }

    ~BackendConfigImpl() override = default;

    const std::string& name() const override {
        return name_;
    }

    const std::set<std::string>& GetSimplifyPreproc() const override {
        return simplify_preproc_;
    }

    const std::set<std::string>& GetSimplify() const override {
        return simplify_;
    }

    bool HasOp(const std::string& op) const override {
        return supported_ops_.count(op) > 0;
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

    void ParseSupportedOps(const json& supported_ops, std::set<std::string>* ops) {
        CHECK(supported_ops.is_object()) << "supported_ops must be an object: " << supported_ops;
        for (const auto& el : supported_ops.items()) {
            CHECK(el.value().is_boolean()) << "op values must be bool: " << supported_ops;
            if (el.value() == false) {
                continue;
            }
            CHECK(ops->emplace(el.key()).second) << "Duplicate key: " << el.key();
        }
    }

    std::string name_;
    std::set<std::string> simplify_preproc_;
    std::set<std::string> simplify_;
    std::set<std::string> supported_ops_;
};

std::unique_ptr<BackendConfig> BackendConfig::FromName(const std::string& name) {
    json j = LoadJSONFromName(name);
    return std::make_unique<BackendConfigImpl>(name, j);
}

std::unique_ptr<BackendConfig> BackendConfig::FromJSON(const std::string& json_str) {
    json j = LoadJSONFromString(json_str);
    return std::make_unique<BackendConfigImpl>("custom", j);
}

}  // namespace chainer_compiler
