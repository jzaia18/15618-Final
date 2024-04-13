#include <cstdlib>
#include <fstream>
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

    // TODO: This loop condition only allows MST, not MSF
    while (n_components > 1) {
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
        }

    }

    return mst;
}

int main(int argc, char **argv) {
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

    std::cout << "Graph on " << n_vertices << " vertices and " << n_edges << " edges" << std::endl;

    // Read all edges from file
    std::vector<Edge> edgelist(n_edges);
    for (int i = 0; i < n_edges; i++) {
        fin >> edgelist[i].u;
        fin >> edgelist[i].v;
        fin >> edgelist[i].weight;
    }

    std::vector<Edge>* mst = boruvka_mst(n_vertices, edgelist);

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
