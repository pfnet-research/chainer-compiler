#pragma once

#include <nlohmann/json.hpp>

namespace chainer_compiler {

using json = nlohmann::json;

json LoadJSONFromName(const std::string& name);

json LoadJSONFromString(const std::string& json_str);

}  // namespace chainer_compiler
