#include "compiler/computation_order/policy_gt.h"

#include <queue>

#include <compiler/flops.h>

typedef std::vector<char> NodeSet;  // Here, I use vector<bool> by taking some risk.

namespace chainer_compiler {

struct SimpleGraph {
  // Simple representation of computational graph
  size_t n;
  std::vector<Value*> value_list;
  std::map<Value*, size_t> value_ids;
  std::vector<std::vector<size_t>> adj;
  std::vector<int64_t> memories;
  std::vector<int64_t> flopses;

  std::string ToString() const {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
      s += "[" + std::to_string(i) + "] => " + value_list[i]->ToString() + "\n";
      s += "      MEM=" + std::to_string(memories[i]) + " FLOPS=" + std::to_string(flopses[i]) + "\n";
    }
    s += "Edges:\n";
    for (size_t i = 0; i < n; ++i) {
      for (const int j : adj[i]) {
        s += " " + std::to_string(i) + " -> " + std::to_string(j) + "\n";
      }
    }
    return s;
  }
};

SimpleGraph GetSimpleFormGraph(const Graph& graph) {
  SimpleGraph sg;

  for (const std::unique_ptr<Value>& value : graph.all_values()) {
    const size_t id = sg.value_ids.size();
    sg.value_ids[value.get()] = id;
    sg.value_list.push_back(value.get());
  }
  sg.n = sg.value_ids.size();

  sg.adj.assign(sg.n, std::vector<size_t>());

  for (Node* node : graph.nodes()) {
    for (Value* input : node->inputs()) {
      for (Value* output : node->outputs()) {
        const size_t in_id = sg.value_ids[input];
        const size_t out_id = sg.value_ids[output];
        sg.adj[in_id].push_back(out_id);
      }
    }
  }

  sg.memories.assign(sg.n, -1);
  for (const std::unique_ptr<Value>& value : graph.all_values()) {
    const size_t id = sg.value_ids[value.get()];
    sg.memories[id] = value->GetNBytes();
  }

  sg.flopses.assign(sg.n, 0);
  for (Node* node : graph.nodes()) {
    for (Value* output : node->outputs()) {
      const size_t out_id = sg.value_ids[output];
      const int64_t f = CalculateFlops(*node);
      sg.flopses[out_id] = f;
    }
  }

  return sg;
}

std::vector<NodeSet> EnumerateLowerSets(const SimpleGraph& sg) {
  std::vector<NodeSet> lower_sets;
  lower_sets.push_back(NodeSet(sg.n, 0));  // Empty node set
  lower_sets.push_back(NodeSet(sg.n, 1));  // Entire node set

  // Run BFS n times
  // reachable[i][j] == 1 iff there is a path from i to j
  std::vector<std::vector<char>> reachable(sg.n, std::vector<char>(sg.n, 0));
  for (size_t i = 0; i < sg.n; ++i) {
    std::queue<size_t> q;
    q.push(i);
    while (!q.empty()) {
      size_t j = q.front();
      q.pop();
      if (reachable[i][j]) continue;
      reachable[i][j] = 1;
      for (size_t k : sg.adj[i]) q.push(k);
    }
  }

  for (size_t j = 0; j < sg.n; ++j) {
    // Add representative lower sets
    NodeSet ls(sg.n);
    for (size_t i = 0; i < sg.n; ++i) ls[i] = reachable[i][j];
    lower_sets.push_back(ls);
  }

  // Sort the lower sets by their size
  std::sort(lower_sets.begin(), lower_sets.end(), [](NodeSet& a, NodeSet& b) {
      size_t asum = 0, bsum = 0;
      for (char e : a) asum += e;
      for (char e : b) bsum += e;
      return asum < bsum;
  });

  return lower_sets;
}

void DiscretizeFlops(SimpleGraph* sg) {
  int64_t low = 1, high = 1LL << 50;
  // Determine the width of discretization by binary-search
  while (high - low > 1) {
    const int64_t width = (high + low) / 2;
    int64_t total = 0;
    for (int64_t flops : sg->flopses) {
      total += (flops + width - 1) / width;
    }
    const int64_t threshold = 5 * static_cast<int64_t>(sg->flopses.size());
    if (total < threshold) {
      high = width;
    } else {
      low = width;
    }
  }

  for (int64_t& flops : sg->flopses) {
    flops = (flops + high - 1) / high;
  }
}

std::vector<NodeSet> ComputeDP(const SimpleGraph& sg, const std::vector<NodeSet>& lower_sets) {
}

std::vector<Order> ComputeOrder(const SimpleGraph& sg, const std::vector<NodesSet>& seq) {
}

std::vector<Order> GTPolicy(const Graph& graph) {
  SimpleGraph sg = GetSimpleFormGraph(graph);

  DiscretizeFlops(&sg);

  std::vector<NodeSet> lower_sets = EnumerateLowerSets(sg);

  std::cout << sg.ToString();

  CHECK(0);

  return {};
}

}  // namespace chainer_compiler
