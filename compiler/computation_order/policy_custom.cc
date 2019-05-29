// Perform the computation with specified order.
// Ther order must be specified in `--custom_computation_order` flag.
// The argument of `--custom_computation_order` must be a comma-separated string that represents a sequence of orders.
// An order should be formatted as either "CF<NODE_ID>", "FF<VALUE_KIND><VALUE_ID>", or "CB<NODE_ID>".
// Here, each of the order formats corresponds to ComputeForward, ForgetForward, and ComputeBackward.
// NODE_ID and VALUE_ID are determined by the index in ONNX file.
// VALUE_KIND is either 'i', 'o', or 't'.
//
// Example: --custom_computation_order CF0,CF1,FFt0,FFo0,CF0,CF1,CB1,CB0

#include "compiler/computation_order/policy_custom.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <compiler/flags.h>
#include <compiler/graph.h>
#include <compiler/node.h>

namespace chainer_compiler {

std::vector<Order> CustomPolicy(const Graph& graph) {
    CHECK(!g_custom_computation_order.empty()) << "--custom_computation_order must be specified.";
    std::stringstream ss(g_custom_computation_order);
    std::string token;

    std::vector<Order> orders;
    while (std::getline(ss, token, ',')) {
        CHECK_GE(token.size(), 3) << "Invalid token in --custom_computation_order: " << token;
        std::string kind = token.substr(0, 2);
        if (kind == "CF") {
            int id = std::stoi(token.substr(2));
            orders.emplace_back(Order::kComputeForward, graph.nodes()[id], nullptr);
        } else if (kind == "FF") {
            Value* value = nullptr;
            char kind = token[2];
            int id = std::stoi(token.substr(3));
            if (kind == 'i') {
                value = graph.input_values()[id];
            } else if (kind == 'o') {
                value = graph.output_values()[id];
            } else if (kind == 't') {
                value = graph.temp_values()[id];
            } else {
                CHECK(false) << "Unknown kind: " << kind;
            }
            orders.emplace_back(Order::kForgetForward, nullptr, value);
        } else if (kind == "BF") {
            int id = std::stoi(token.substr(2));
            orders.emplace_back(Order::kComputeBackward, graph.nodes()[id], nullptr);
        } else {
            CHECK(false) << "Unknown format in token: " << kind;
        }
    }
    return orders;
}

}  // namespace chainer_compiler
