#include "type_inference.h"

#include <compiler/dtype_inference.h>
#include <compiler/graph.h>

namespace chainer_compiler {

void InferDtypeAndShape(Node* node) {
    InferDtype(node);
}

void InferAllDtypeAndShape(Graph* graph) {
    for (Node* node : graph->GetTopologicallySortedNodes()) {
        InferDtypeAndShape(node);
    }
}

}  // namespace chainer_compiler
