#include "boruvkas.h"

#include <iostream>
#include <string>

#include "parlaylib/examples/helper/graph_utils.h"
#include "parlaylib/include/parlay/internal/get_time.h"
#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/sequence.h"

// **************************************************************
// Driver
// **************************************************************
using vertex = int;
using wtype = int;
using weighted_graph =
    parlay::sequence<parlay::sequence<std::pair<vertex, wtype>>>;

// static weighted_graph read_graph_from_file_custom(const std::string& filename) {
//     auto str = parlay::file_map(filename);
//     auto tokens = parlay::tokens(str);
//     long n = parlay::chars_to_long(tokens[0]);
//     long m = parlay::chars_to_long(tokens[1]);
//     if (tokens.size() != m * 3 + 2) {
//         std::cout << "bad file format" << std::endl;
//         exit(1);
//     }
//     weighted_graph G = parlay::tabulate(n, [&](long i) {
//         return parlay::sequence<std::pair<vertex, wtype>>();
//     });

//     for (int i = 0; i < m; i++) {
//         long u = parlay::chars_to_long(tokens[3 * i + 2]);
//         long v = parlay::chars_to_long(tokens[3 * i + 3]);
//         wtype w = parlay::chars_to_long(tokens[3 * i + 4]);
//         G[u].push_back(std::make_pair(v, w));
//         G[v].push_back(std::make_pair(u, w));
//     }
//     return G;
// }



int main(int argc, char* argv[]) {
    auto usage = "Usage: boruvka <filename>";
    if (argc != 2) {
        std::cout << usage << std::endl;
        exit(1);
    }

    auto str = parlay::file_map(argv[1]);
    auto tokens = parlay::tokens(str);
    long n = parlay::chars_to_long(tokens[0]);
    long m = parlay::chars_to_long(tokens[1]);
    if (tokens.size() != m * 3 + 2) {
        std::cout << "bad file format" << std::endl;
        exit(1);
    }
    edges<vertex, wtype> E;

    for (int i = 0; i < m; i++) {
        long u = parlay::chars_to_long(tokens[3 * i + 2]);
        long v = parlay::chars_to_long(tokens[3 * i + 3]);
        wtype w = parlay::chars_to_long(tokens[3 * i + 4]);
        E.push_back(std::make_tuple(u, v, w));
    }

    parlay::sequence<vertex> result;
    parlay::internal::timer t("Time");
    for (int i = 0; i < 3; i++) {
        boruvka<vertex, wtype>(E, n);
        t.next("boruvka");
    }
}