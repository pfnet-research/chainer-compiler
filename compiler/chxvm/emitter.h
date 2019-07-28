#pragma once

#include <iosfwd>
#include <vector>

namespace chainer_compiler {

class Graph;
class Model;
class Node;
class Value;

namespace runtime {
class ChxVMProgramProto;
}

namespace chxvm {

void Emit(const Model& model, runtime::ChxVMProgramProto* program, bool dump_value_names = false);

void Emit(const Graph& graph, runtime::ChxVMProgramProto* program, bool dump_value_names = false);

void Emit(const Model& model, std::ostream& out, bool dump_value_names = false);

void Emit(
        const std::vector<Node*>& nodes,
        const std::vector<Value*>& feeds,
        const std::vector<Value*>& fetches,
        runtime::ChxVMProgramProto* program,
        std::vector<int>* input_ids,
        std::vector<int>* output_ids);

}  // namespace chxvm
}  // namespace chainer_compiler
