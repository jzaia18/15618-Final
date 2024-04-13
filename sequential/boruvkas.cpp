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

    // TODO: This only allows MST, not MSF
    while (n_components > 1) {
        // initialize components
        for (int i = 0; i < n_vertices; i++) {
            vertices[i].cheapest_edge = nullptr;
        }

        for (const Edge& e : edgelist) {
            // Skip edges that connect a component to itself
            if (get_component(vertices, e.u) == get_component(vertices, e.v)) {
                continue;
            }

            // Check if this edge is the cheapest (so far) for its vertices
            if (vertices[e.u].cheapest_edge == nullptr or e < *vertices[e.u].cheapest_edge) {
                vertices[e.u].cheapest_edge = &e;
            }
            if (vertices[e.v].cheapest_edge == nullptr or e < *vertices[e.v].cheapest_edge) {
                vertices[e.v].cheapest_edge = &e;
            }
        }

        // for (int i = 0; i < n_vertices; i++) {
        //     if (vertices[i].cheapest_edge != nullptr && vertices[i].component != i) {
        //         int component = get_component(vertices, i);
        //         if (vertices[i].cheapest_edge < vertices[component].cheapest_edge) {
        //             vertices[component].cheapest_edge = vertices[i].cheapest_edge;
        //         }

        //         vertices[i].cheapest_edge = nullptr;
        //     }
        // }

        // Connect newest edges to MST
        for (int i = 0; i < n_vertices; i++) {
            if (vertices[i].cheapest_edge != nullptr) {
                if (get_component(vertices, vertices[i].cheapest_edge->u) != get_component(vertices, vertices[i].cheapest_edge->v)) {
                    mst->push_back(*vertices[i].cheapest_edge);
                    merge_components(vertices, vertices[i].cheapest_edge->u, vertices[i].cheapest_edge->v);
                    n_components--;
                }
            }
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
    for (const Edge& e : *mst) {
        weight += e.weight;
        if (verbose) {
            std::cout << "(" << e.u << ", " << e.v << " | " << e.weight << ")  ";
        }
    }
    std::cout << std::endl << "Total weight: " << weight << std::endl;

    delete mst;

    return EXIT_SUCCESS;
}
