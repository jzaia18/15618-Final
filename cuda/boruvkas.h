#include <unistd.h>

struct Edge {
    int u;
    int v;
    int weight;
};

struct Vertex {
    // the index of the vertex is implied since vertices are stored in a vector
    // int index;
    int component;
    const Edge* cheapest_edge;
};

struct MST {
    Edge* mst;
    size_t size;
    size_t capacity;
    int weight;
};

MST boruvka_mst(const int n_vertices, const int n_edges, const Edge* edgelist);
