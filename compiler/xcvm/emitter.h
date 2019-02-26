#pragma once

#include <iosfwd>
#include <vector>

namespace chainer_compiler {

class Graph;
class Model;
class Node;
class Value;

namespace runtime {
class XCProgramProto;
}

namespace xcvm {

void Emit(const Model& model, runtime::XCProgramProto* program, bool dump_value_names = false);

void Emit(const Graph& graph, runtime::XCProgramProto* program, bool dump_value_names = false);

void Emit(const Model& model, std::ostream& out, bool dump_value_names = false);

void Emit(
        const std::vector<Node*>& nodes,
        const std::vector<Value*>& feeds,
        const std::vector<Value*>& fetches,
        runtime::XCProgramProto* program,
        std::vector<int>* input_ids,
        std::vector<int>* output_ids);

}  // namespace xcvm
}  // namespace chainer_compiler
