#include <unistd.h>

typedef unsigned long long ullong ;

struct __attribute__((packed)) Edge {
    uint u;
    uint v;
    uint weight;
};

struct Vertex {
    // the index of the vertex is implied since vertices are stored in a vector
    // int index;
    ullong component;
    ullong cheapest_edge;
};

struct MST {
    int* mst; // stores indices of chosen edges
    ullong weight;
};

MST boruvka_mst(const ullong n_vertices, const ullong n_edges, const Edge* edgelist);
void initGPUs();

// inline int edge_cmp(const void *_lhs, const void *_rhs)
// {
//     const Edge& lhs = *((const Edge*)_lhs);
//     const Edge& rhs = *((const Edge*)_rhs);
//     const int sub = lhs.weight - rhs.weight;
//     if (sub) {
//         return sub;
//     }
//     if (lhs.u == rhs.u) {
//         return (lhs.v - rhs.v);
//     }
//     return (lhs.u - rhs.u);
// }
