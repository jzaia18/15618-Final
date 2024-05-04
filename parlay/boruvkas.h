#include <atomic>

#include "parlaylib/examples/helper/union_find.h"
#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/sequence.h"

template <typename vertex, typename wtype>
using edge = std::tuple<vertex, vertex, wtype>;

template <typename vertex, typename wtype>
auto boruvka(const parlay::sequence<edge<vertex, wtype>>& E, vertex n) {
    auto curr_edges = E;
    union_find<vertex> UF(n);
    auto sol = parlay::tabulate<edge<vertex, wtype>>(
        0, [](long i) { return std::make_tuple(-1, -1, -1); });

    while (curr_edges.size() > 0) {
        auto best_edge_index =
            parlay::tabulate<std::atomic<int>>(n, [](long i) { return -1; });
        parlay::parallel_for(0, curr_edges.size(), [&](int i) {
            auto [u, v, w] = curr_edges[i];

            int index = best_edge_index[u].load();
            while ((index == -1 || w < std::get<2>(curr_edges[index])) &&
                   (best_edge_index[u].load() != index ||
                    best_edge_index[u].compare_exchange_strong(index, i))) {
                index = best_edge_index[u].load();
            }

            index = best_edge_index[v].load();
            while ((index == -1 || w < std::get<2>(curr_edges[index])) &&
                   (best_edge_index[v].load() != index ||
                    best_edge_index[v].compare_exchange_strong(index, i))) {
                index = best_edge_index[v].load();
            }
        });

        sol = parlay::append(
            sol, parlay::map_maybe(parlay::iota(n), [&](int i) {
                int edge_index = best_edge_index[i];
                if (edge_index == -1)
                    return std::optional<edge<vertex, wtype>>{};

                auto [u, v, w] = curr_edges[edge_index];
                // If this edge is covered twice, only union when i == u (u < v
                // naturally)
                if (i == v && edge_index == best_edge_index[u])
                    return std::optional<edge<vertex, wtype>>{};

                // UF.link(UF.find(u), v);
                bool done = false;
                vertex root = -1;
                auto un = u;
                while (!done) {
                    un = UF.find(un);
                    done = UF.parents[un] == root &&
                           UF.parents[un].compare_exchange_strong(root, v);
                }

                // add edge
                return std::optional{curr_edges[edge_index]};
            }));

        curr_edges =
            parlay::map_maybe(curr_edges, [&](const edge<vertex, wtype>& x) {
                auto [u, v, w] = x;
                vertex un = UF.find(u);
                vertex vn = UF.find(v);
                return (un == vn) ? std::optional<edge<vertex, wtype>>{}
                                  : std::optional{std::make_tuple(un, vn, w)};
            });
    }

    return sol;
}
