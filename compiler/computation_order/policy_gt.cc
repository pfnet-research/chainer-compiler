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

bool IsIncreasing(const NodeSet& ls, const NodeSet& ls_next) {
  return true;
}

std::tuple<int64_t, int64_t, size_t>
ComputeConsumptionInfo(const SimpleGraph& sg, const NodeSet& ls, const NodeSet& ls_next) {
  return std::make_tuple(0L, 0L, 0UL);
}

std::vector<NodeSet> ComputeDP(const SimpleGraph& sg, const std::vector<NodeSet>& lower_sets, const size_t& budget) {
  size_t nl = lower_sets.size();
  // opt[lower_set_index][flops] := <minimum memory consumption, prev_ls, prev_flops>
  std::map<size_t, std::tuple<int64_t, size_t, size_t>> opt[nl];
  opt[0][0] = std::make_tuple(0, nl + 1, 0);

  for (size_t i = 0; i < nl; ++i) {
    const NodeSet& ls = lower_sets[i];
    for (size_t i_next = 0; i_next < nl; ++i_next) {
      const NodeSet& ls_next = lower_sets[i_next];
      if (!IsIncreasing(ls, ls_next)) continue;

      const auto info = ComputeConsumptionInfo(sg, ls, ls_next);
      const int64_t additional_memory1 = std::get<0>(info);
      const int64_t additional_memory234 = std::get<1>(info);
      const size_t additional_flops = std::get<2>(info);

      for (const auto& p : opt[i]) {
        const size_t flops = p.first;
        const int64_t memopt = std::get<0>(p.second);
        const int64_t total_memory = memopt + additional_memory234;
        if (total_memory > budget) continue;

        const size_t flops_next = flops + additional_flops;
        const int64_t memopt_next = memopt + additional_memory1;

        auto next_opt_it = opt[i_next].find(flops_next);

        if (next_opt_it == opt[i_next].end()) {
          opt[i_next].insert({flops_next, std::make_tuple(memopt_next, i, flops)});
        } else if (std::get<0>(next_opt_it->second) > memopt_next) {
          next_opt_it->second = std::make_tuple(memopt_next, i, flops);
        }
      }
    }
  }

  std::vector<NodeSet> seq;
  const auto last_it = opt[nl - 1].begin();
  if (last_it != opt[nl - 1].end()) {
    size_t i = nl - 1;
    size_t flops = last_it->first;

    while (i < nl) {
      seq.push_back(lower_sets[i]);
      const auto it = opt[i].find(flops);
      i = std::get<1>(it->second);
      flops = std::get<2>(it->second);
    }
    std::reverse(seq.begin(), seq.end());
  }
  return seq;
}

std::vector<Order> ComputeOrder(const SimpleGraph& sg, const std::vector<NodeSet>& seq) {
  return {};
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
