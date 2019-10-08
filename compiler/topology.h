#pragma once

#include <vector>

namespace chainer_compiler {

class Node;
class Value;

// Classifies input and output values of all `nodes` to one of three
// kinds, i.e., `inputs`, `outputs`, and `temps`.
//
// 1. A value is an input when no `nodes` output the value.
// 2. A value is an output when there is a consumer of the value
//    outside `nodes`.
// 3. A value is temporary otherwise.
void ClassifyValues(const std::vector<Node*>& nodes, std::vector<Value*>* inputs, std::vector<Value*>* outputs, std::vector<Value*>* temps);

// Returns `nodes` after sorting it topologically. Nodes which is
// unreachable from `inputs` will be discarded.
std::vector<Node*> SortTopologically(const std::vector<Node*>& nodes, const std::vector<Value*>& inputs, bool is_full_graph);

// Returns `nodes` and their distances from `inputs` after sorting it
// topologically. Nodes which is unreachable from `inputs` will be discarded.
std::vector<std::pair<Node*, int>> SortTopologicallyWithDistance(
        const std::vector<Node*>& nodes, const std::vector<Value*>& inputs, bool is_full_graph);

// Returns values related to `nodes` and their distances from `inputs` after
// sorting it topologically. Nodes which is unreachable from `inputs` will be
// discarded.
std::vector<std::pair<Value*, int>> SortValuesTopologicallyWithDistance(
        const std::vector<Node*>& nodes, const std::vector<Value*>& inputs, bool is_full_graph);

}  // namespace chainer_compiler
