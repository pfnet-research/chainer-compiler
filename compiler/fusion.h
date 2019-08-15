#pragma once

#include <functional>
#include <set>
#include <string>

namespace chainer_compiler {

class Graph;
class Node;

void FuseOperations(Graph* graph);

void CreateFusionGroup(
        Graph* graph, const std::set<Node*>& nodes, const std::string& fusion_type, int fusion_group_id, bool can_fuse_initializers);

void FuseAllConnectedNodes(
        const char* name, Graph* graph, int min_fuse_ops, bool can_fuse_initializers, const std::function<bool(const Node&)>& is_fusable);

void FuseDldtOperations(Graph* graph);
void FuseNGraphOperations(Graph* graph);
void FuseTVMOperations(Graph* graph);
void FuseSNPEOperations(Graph* graph);
void FuseElementwiseOperations(Graph* graph);

}  // namespace chainer_compiler
