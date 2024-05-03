#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "boruvkas.h"

inline int get_component(std::vector<Vertex>& componentlist, const int i) {
    int curr = componentlist[i].component;

    while (componentlist[curr].component != curr) {
        curr = componentlist[curr].component;
    }

    componentlist[i].component = curr;
    return curr;
}

inline void merge_components(std::vector<Vertex>& componentlist, const int i, const int j) {
    componentlist[get_component(componentlist, i)].component = get_component(componentlist, j);
}

std::vector<Edge>* boruvka_mst(int n_vertices, const std::vector<Edge>& edgelist) {
    std::vector<Edge>* mst = new std::vector<Edge>();
    std::vector<Vertex> vertices(n_vertices);

    // initialize components
    for (int i = 0; i < n_vertices; i++) {
        vertices[i] = Vertex{i, nullptr}; //Vertex{i, i, nullptr};
    }

    int n_components = n_vertices;
    bool keep_going;

    do {
        keep_going = false;
        for (const Edge& e : edgelist) {
            int c1 = get_component(vertices, e.u);
            int c2 = get_component(vertices, e.v);

            // Skip edges that connect a component to itself
            if (c1 == c2) {
                continue;
            }

            // Check if this edge is the cheapest (so far) for its connected components
            if (vertices[c1].cheapest_edge == nullptr || e < *vertices[c1].cheapest_edge) {
                vertices[c1].cheapest_edge = &e;
            }
            if (vertices[c2].cheapest_edge == nullptr || e < *vertices[c2].cheapest_edge) {
                vertices[c2].cheapest_edge = &e;
            }
        }

        keep_going = false;
        // Connect newest edges to MST
        for (int i = 0; i < n_vertices; i++) {
            const Edge* edge_ptr = vertices[i].cheapest_edge;
            if (edge_ptr == nullptr) {
                continue;
            }

            // if (get_component(vertices, edge_ptr->u) == get_component(vertices, edge_ptr->v)) {
            //     continue;
            // }

            mst->push_back(*edge_ptr);
            vertices[get_component(vertices, edge_ptr->u)].cheapest_edge = nullptr;
            vertices[get_component(vertices, edge_ptr->v)].cheapest_edge = nullptr;
            merge_components(vertices, edge_ptr->u, edge_ptr->v);
            n_components--;
            keep_going = true;
        }

    } while (keep_going && n_components > 1);

    return mst;
}

int main(int argc, char **argv) {
    const auto init_start = std::chrono::steady_clock::now();
    int n_vertices;
    int n_edges;
    std::string input_filename;
    bool verbose = false;
    bool bin = false;

    int opt;
    while ((opt = getopt(argc, argv, "f:abv")) != -1) {
        switch (opt) {
            case 'f':
                input_filename = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 'a':
                bin = false;
                break;
            case 'b':
                bin = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f [-a] [-b] input_filename [-v]\n";
                exit(EXIT_FAILURE);
        }
    }

    std::vector<Edge> edgelist;

    if (bin) {
        std::ifstream fin(input_filename, std::ios::binary);

        if (!fin) {
            std::cerr << "Unable to open file: " << input_filename << std::endl;
            exit(EXIT_FAILURE);
        }

        fin.read((char*)&n_vertices, sizeof(int));
        fin.read((char*)&n_edges, sizeof(int));

        std::cout << "Reading graph on " << n_vertices << " vertices and " << n_edges << " edges..." << std::endl;

        // Read all edges from file
        edgelist = std::vector<Edge>(n_edges);
        for (int i = 0; i < n_edges; i++) {
            fin.read((char*)&edgelist[i], 3 * sizeof(int));
        }
    } else {
        std::ifstream fin(input_filename);
        if (!fin) {
            std::cerr << "Unable to open file: " << input_filename << std::endl;
            exit(EXIT_FAILURE);
        }

        fin >> n_vertices;
        fin >> n_edges;

        std::cout << "Reading graph on " << n_vertices << " vertices and " << n_edges << " edges..." << std::endl;

        // Read all edges from file
        edgelist = std::vector<Edge>(n_edges);
        for (int i = 0; i < n_edges; i++) {
            fin >> edgelist[i].u;
            fin >> edgelist[i].v;
            fin >> edgelist[i].weight;
        }
    }

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

    std::vector<Edge>* mst = boruvka_mst(n_vertices, edgelist);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    int weight = 0;
    if (verbose) std::cout << "[";
    for (const Edge& e : *mst) {
        weight += e.weight;
        if (verbose) {
            std::cout << "(" << e.u << ", " << e.v << ", " << e.weight << "), ";
        }
    }
    if (verbose) {
        std::cout << "]" << std::endl;
    }
    std::cout << "Total weight: " << weight << std::endl;

    delete mst;

    return EXIT_SUCCESS;
}
