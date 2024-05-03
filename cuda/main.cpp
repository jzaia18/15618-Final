#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <unistd.h>

#include "boruvkas.h"

int main(int argc, char **argv){

    const auto init_start = std::chrono::steady_clock::now();
    int n_vertices;
    int n_edges;
    std::string input_filename;
    bool verbose = false;

    int opt;
    while ((opt = getopt(argc, argv, "f:v")) != -1) {
        switch (opt) {
            case 'f':
                input_filename = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename [-v]\n";
                exit(EXIT_FAILURE);
        }
    }

    std::ifstream fin(input_filename);
    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    fin >> n_vertices;
    fin >> n_edges;

    std::cout << "Reading graph on " << n_vertices << " vertices and " << n_edges << " edges..." << std::endl;

    // Read all edges from file
    Edge* edgelist = (Edge*) malloc(n_edges * sizeof(Edge));
    for (int i = 0; i < n_edges; i++) {
        fin >> edgelist[i].u;
        fin >> edgelist[i].v;
        fin >> edgelist[i].weight;
    }

    // Preprocess edges by sorting
    qsort(edgelist, n_edges, sizeof(Edge), edge_cmp);

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

    MST result = boruvka_mst(n_vertices, n_edges, edgelist);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    if (verbose) {
        std::cout << "[";
        for (int i = 0; i < result.size; i++) {
            const Edge& e = edgelist[result.mst[i]];
            std::cout << "(" << e.u << ", " << e.v << ", " << e.weight << "), ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "Total weight: " << result.weight << std::endl;

    free(edgelist);
    free(result.mst);

    return EXIT_SUCCESS;
}
