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
    int cheapest_edge;
};

struct MST {
    int* mst; // stores indices of chosen edges
    size_t size;
    size_t capacity;
    int weight;
};

MST boruvka_mst(const int n_vertices, const int n_edges, const Edge* edgelist);

inline int qsort_edge_cmp(const void *_lhs, const void *_rhs)
{
    const Edge& lhs = *((const Edge*)_lhs);
    const Edge& rhs = *((const Edge*)_rhs);
    const int sub = lhs.weight - rhs.weight;
    if (sub) {
        return sub;
    }
    if (lhs.u == rhs.u) {
        return (lhs.v - rhs.v);
    }
    return (lhs.u - rhs.u);
}
