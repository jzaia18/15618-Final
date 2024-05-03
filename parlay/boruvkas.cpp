#include "boruvkas.h"

#include <unistd.h>
#include <chrono>
#include <iostream>
#include <string>

#include "parlaylib/examples/helper/graph_utils.h"
#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/sequence.h"

// **************************************************************
// Driver
// **************************************************************
using vertex = int;
using wtype = uint;

int main(int argc, char* argv[]) {
    const auto init_start = std::chrono::steady_clock::now();

    std::string input_filename;
    bool bin = false;
    uint reps = 1;

    int opt;
    while ((opt = getopt(argc, argv, "f:abr:")) != -1) {
        switch (opt) {
            case 'f':
                input_filename = optarg;
                break;
            case 'r':
                reps = strtol(optarg, NULL, 10);
                break;
            case 'a':
                bin = false;
                break;
            case 'b':
                bin = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-a] [-b] -f input_filename\n";
                exit(EXIT_FAILURE);
        }
    }

    uint n, m;
    parlay::sequence<edge<vertex, wtype>> E;
    if (bin) {
        std::ifstream fin(input_filename, std::ios::binary);

        if (!fin) {
            std::cerr << "Unable to open file: " << input_filename << std::endl;
            exit(EXIT_FAILURE);
        }

        fin.read((char*)&n, sizeof(int));
        fin.read((char*)&m, sizeof(int));

        std::cout << "Reading graph on " << n << " vertices and " << m << " edges..." << std::endl;

        // Read all edges from file
        vertex u;
        vertex v;
        wtype w;
        for (int i = 0; i < m; i++) {
            fin.read((char*)&u, sizeof(int));
            fin.read((char*)&v, sizeof(int));
            fin.read((char*)&w, sizeof(unsigned int));
            E.push_back(std::make_tuple(u, v, w));
        }
    } else {
        auto str = parlay::file_map(input_filename);
        auto tokens = parlay::tokens(str);

        n = parlay::chars_to_uint(tokens[0]);
        m = parlay::chars_to_uint(tokens[1]);

        if (tokens.size() != m * 3 + 2) {
            std::cout << "bad file format" << std::endl;
            exit(1);
        }

        std::cout << "Reading graph on " << n << " vertices and " << m << " edges..." << std::endl;

        for (int i = 0; i < m; i++) {
            vertex u = parlay::chars_to_int(tokens[3 * i + 2]);
            vertex v = parlay::chars_to_int(tokens[3 * i + 3]);
            wtype w = parlay::chars_to_uint(tokens[3 * i + 4]);
            E.push_back(std::make_tuple(u, v, w));
        }
    }

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    parlay::sequence<vertex> result;
    for (uint i = 0; i < reps; i++) {
        const auto compute_start = std::chrono::steady_clock::now();
        auto sol = boruvka<vertex, wtype>(E, n);

        const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        std::cout << "Computation time (sec): " << compute_time << '\n';

        int total_weight = 0;
        for (auto [u, v, w] : sol) {
            total_weight += w;
        }

        std::cout << "Total weight: " << total_weight << std::endl;
    }
}
