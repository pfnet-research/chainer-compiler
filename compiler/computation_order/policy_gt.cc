#include "compiler/computation_order/policy_gt.h"

#include <queue>

#include <compiler/flags.h>
#include <compiler/flops.h>

typedef std::vector<char> NodeSet;  // Here, I use vector<bool> by taking some risk.

namespace chainer_compiler {

struct SimpleGraph {
  // Simple representation of computational graph
  size_t n;
  std::vector<Value*> value_list;
  std::map<Value*, size_t> value_ids;
  std::vector<std::vector<size_t>> adj;  // adj[i] is a list of vertices adjacent from the vertex i
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

bool IsIncreasing(const NodeSet& ls, const NodeSet& ls_next) {
  // Check if ls < ls_next.
  for (size_t i = 0; i < ls.size(); ++i) {
    if (ls[i] > ls_next[i]) return false;
  }
  return true;
}

NodeSet SetMinus(const NodeSet& ls1, const NodeSet& ls2) {
  // Compute (L1 \setminus L2)
  const size_t n = ls1.size();
  NodeSet ret(n);
  for (size_t i = 0; i < n; ++i) ret[i] = ls1[i] && !ls2[i];
  return ret;
}

NodeSet DeltaPlus(const SimpleGraph& sg, const NodeSet& ls) {
  const size_t n = ls.size();
  NodeSet ret(n);
  for (size_t i = 0; i < n; ++i) {
    if (ls[i]) {
      for (size_t j : sg.adj[i]) ret[j] = 1;
    }
  }
  return ret;
}

NodeSet DeltaMinus(const SimpleGraph& sg, const NodeSet& ls) {
  const size_t n = ls.size();
  NodeSet ret(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j : sg.adj[i]) {
      if (ls[j]) ret[i] = 1;
    }
  }
  return ret;
}

NodeSet Boundary(const SimpleGraph& sg, const NodeSet& ls) {
  const size_t n = ls.size();
  NodeSet ret(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j : sg.adj[i]) {
      if (!ls[j] && ls[i]) ret[i] = 1;
    }
  }
  return ret;
}

SimpleGraph GetSimpleFormGraph(const Graph& graph) {
  // Extract only temporary&output values
  SimpleGraph sg;

  std::vector<Value*> intermediate_values(graph.temp_values());
  for (Value* output : graph.output_values()) {
    intermediate_values.push_back(output);
  }

  for (Value* value : intermediate_values) {
    const size_t id = sg.value_ids.size();
    sg.value_ids[value] = id;
    sg.value_list.push_back(value);
  }
  sg.n = sg.value_ids.size();

  sg.adj.assign(sg.n, std::vector<size_t>());

  for (Node* node : graph.nodes()) {
    for (Value* input : node->inputs()) {
      for (Value* output : node->outputs()) {
        const auto in_it = sg.value_ids.find(input);
        const auto out_it = sg.value_ids.find(output);
        if (in_it != sg.value_ids.end() && out_it != sg.value_ids.end()) {
          sg.adj[in_it->second].push_back(out_it->second);
        }
      }
    }
  }

  sg.memories.assign(sg.n, -1);
  for (Value* value : intermediate_values) {
    const size_t id = sg.value_ids[value];
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
      for (size_t k : sg.adj[j]) q.push(k);
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

std::tuple<int64_t, int64_t, size_t>
ComputeConsumptionInfo(const SimpleGraph& sg, const NodeSet& ls, const NodeSet& ls_next) {
  // Returns {MemoryCost1, MemoryCost2+3+4, FlopsCost}.
  const NodeSet boundary = Boundary(sg, ls_next);
  const NodeSet vs = SetMinus(ls_next, ls);
  const NodeSet deltaplus = DeltaPlus(sg, ls_next);

  const NodeSet flops_set = SetMinus(vs, boundary);
  const NodeSet memory_set1 = SetMinus(boundary, ls);
  const NodeSet memory_set3 = SetMinus(deltaplus, ls_next);
  const NodeSet memory_set4 = SetMinus(DeltaMinus(sg, deltaplus), ls_next);

  auto compute_cost = [](const NodeSet& set, const std::vector<int64_t>& costs) {
    int64_t total = 0;
    for (size_t i = 0; i < set.size(); ++i) {
      if (set[i]) total += costs[i];
    }
    return total;
  };

  const int64_t flops = compute_cost(flops_set, sg.flopses);
  const int64_t mem1 = compute_cost(memory_set1, sg.memories);
  const int64_t mem234 = 2 * compute_cost(vs, sg.memories) + compute_cost(memory_set3, sg.memories) + compute_cost(memory_set4, sg.memories);

  return std::make_tuple(mem1, mem234, flops);
}

std::vector<NodeSet> ComputeDP(const SimpleGraph& sg, const std::vector<NodeSet>& lower_sets, const int64_t& budget) {
  size_t nl = lower_sets.size();
  // opt[lower_set_index][flops] := <minimum memory consumption, prev_ls, prev_flops>
  std::map<size_t, std::tuple<int64_t, size_t, size_t>> opt[nl];
  opt[0][0] = std::make_tuple(0, nl + 1, 0);

  for (size_t i = 0; i < nl; ++i) {
    // std::cout << "#SIZE[" << i << "]=" << opt[i].size() << std::endl;
    // if (opt[i].size()) {
      // std::cout << "   " << opt[i].begin()->first << " " << opt[i].rbegin()->first << std::endl;
      // std::cout << "   " << std::get<0>(opt[i].begin()->second) / 1000000 << " MB" << std::endl;
    // }
    const NodeSet& ls = lower_sets[i];
    for (size_t i_next = i + 1; i_next < nl; ++i_next) {
      const NodeSet& ls_next = lower_sets[i_next];
      if (!IsIncreasing(ls, ls_next)) continue;

      const auto info = ComputeConsumptionInfo(sg, ls, ls_next);
      const int64_t additional_memory1 = std::get<0>(info);
      const int64_t additional_memory234 = std::get<1>(info);
      const size_t additional_flops = std::get<2>(info);

      // std::cout << "#### = " << (double)(additional_memory1 / 1000) / 1000 << "MB, " << additional_memory234 / 1000000 << "MB" << std::endl;

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
  const int64_t budget = g_gt_budget * 1000000LL;
  SimpleGraph sg = GetSimpleFormGraph(graph);

  DiscretizeFlops(&sg);

  const std::vector<NodeSet> lower_sets = EnumerateLowerSets(sg);
  for (auto ls : lower_sets) {
    int t = 0;
    for (char c : ls) t += c;
    std::cout << t << std::endl;
  }

  const std::vector<NodeSet> seq = ComputeDP(sg, lower_sets, budget);

  std::cout << "#=" << seq.size() << std::endl;

  CHECK(0);

  return {};
}

}  // namespace chainer_compiler
