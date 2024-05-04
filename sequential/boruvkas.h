#include <limits.h>

#define NO_EDGE (INT_MAX)

struct __attribute__((packed)) Edge {
    uint u;
    uint v;
    uint weight;

    // Define operators so that edges can be compared
    bool operator <(const Edge& rhs) const
    {
        if (this->weight < rhs.weight) {
            return true;
        }
        if (this->weight == rhs.weight) {
            if (this->u == rhs.u) {
                return (this->v < rhs.v);
            }
            return (this->u < rhs.u);
        }
        return false;
    }
    bool operator >(const Edge& rhs) const
    {
        if (this->weight > rhs.weight) {
            return true;
        }
        if (this->weight == rhs.weight) {
            if (this->u == rhs.u) {
                return (this->v > rhs.v);
            }
            return (this->u > rhs.u);
        }
        return false;
    }
    // bool operator ==(const Edge& rhs) const
    // {
    //     return this->u == rhs.u && this->v == rhs.v && this->weight == rhs.weight;
    // }
};

struct Vertex {
    // the index of the vertex is implied since vertices are stored in a vector
    // int index;
    ulong component;
    ulong cheapest_edge; // the index of the cheapest edge
    // const Edge* cheapest_edge;
};
