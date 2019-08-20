#pragma once

namespace chainer_compiler {

class Node;

namespace runtime {
class ChxVMProgramProto;
}

namespace chxvm {

class ValueIdManager;

void EmitSimpleNode(const Node& node, const ValueIdManager& id_manager, runtime::ChxVMProgramProto* prog);

}  // namespace chxvm
}  // namespace chainer_compiler
