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
using wtype = uint;

int main(int argc, char* argv[]) {
    auto usage = "Usage: boruvkas <filename>";
    if (argc != 2) {
        std::cout << usage << std::endl;
        exit(1);
    }

    auto str = parlay::file_map(argv[1]);
    auto tokens = parlay::tokens(str);
    uint n = parlay::chars_to_uint(tokens[0]);
    uint m = parlay::chars_to_uint(tokens[1]);
    if (tokens.size() != m * 3 + 2) {
        std::cout << "bad file format" << std::endl;
        exit(1);
    }
    parlay::sequence<edge<vertex, wtype>> E;

    for (int i = 0; i < m; i++) {
        vertex u = parlay::chars_to_int(tokens[3 * i + 2]);
        vertex v = parlay::chars_to_int(tokens[3 * i + 3]);
        wtype w = parlay::chars_to_uint(tokens[3 * i + 4]);
        E.push_back(std::make_tuple(u, v, w));
    }

    parlay::sequence<vertex> result;
    parlay::internal::timer t("Time");
    for (int i = 0; i < 3; i++) {
        boruvka<vertex, wtype>(E, n);
        t.next("boruvka");
    }
}