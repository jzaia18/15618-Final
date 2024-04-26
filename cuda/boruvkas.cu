#include <chrono>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include "boruvkas.h"

// Define constants for CUDA threadblocks
#define THREADBLOCK_WIDTH (8)
#define THREADBLOCK_HEIGHT (4)
#define BLOCKSIZE (THREADBLOCK_WIDTH*THREADBLOCK_HEIGHT)

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {
    // TODO
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;


// Define operators so that edges can be compared
__device__ inline int edge_cmp(const Edge& lhs, const Edge& rhs)
{
    const int sub = lhs.weight - rhs.weight;
    if (sub) {
        return sub;
    }
    if (lhs.u == rhs.u) {
        return (lhs.v - rhs.v);
    }
    return (lhs.u - rhs.u);
}

__device__ inline int get_component(Vertex* componentlist, const int i) {
    int curr = componentlist[i].component;

    while (componentlist[curr].component != curr) {
        curr = componentlist[curr].component;
    }

    componentlist[i].component = curr;
    return curr;
}

__device__ inline void merge_components(Vertex* componentlist, const int i, const int j) {
    componentlist[get_component(componentlist, i)].component = get_component(componentlist, j);
}

__device__ inline void assign_cheapest(Vertex* vertices, const Edge* edgelist, const int n_edges, const int threadID) {
    const int start = (threadID * n_edges / BLOCKSIZE);
    const int end = ((threadID+1) * n_edges / BLOCKSIZE);

    for (int i = start; i < end; i++) {
        const Edge& e = edgelist[i];
        int c1 = get_component(vertices, e.u);
        int c2 = get_component(vertices, e.v);

        // Skip edges that connect a component to itself
        if (c1 == c2) {
            continue;
        }

        // Check if this edge is the cheapest (so far) for its connected components
        if (vertices[c1].cheapest_edge == nullptr || edge_cmp(e, *vertices[c1].cheapest_edge) < 0) {
            vertices[c1].cheapest_edge = &e;
        }
        if (vertices[c2].cheapest_edge == nullptr || edge_cmp(e, *vertices[c2].cheapest_edge) < 0) {
            vertices[c2].cheapest_edge = &e;
        }
    }
}

__device__ inline int update_mst(Vertex* vertices, const Edge* edgelist, MST* mst, const int n_vertices, const int n_edges) {
    int n_unions = 0;
    // Connect newest edges to MST
    for (int i = 0; i < n_vertices; i++) {
        const Edge* edge_ptr = vertices[i].cheapest_edge;
        if (edge_ptr == nullptr) {
            continue;
        }

        // if (get_component(vertices, edge_ptr->u) == get_component(vertices, edge_ptr->v)) {
        //     continue;
        // }

        mst->mst[mst->size++] = *edge_ptr;
        vertices[get_component(vertices, edge_ptr->u)].cheapest_edge = nullptr;
        vertices[get_component(vertices, edge_ptr->v)].cheapest_edge = nullptr;
        merge_components(vertices, edge_ptr->u, edge_ptr->v);
        n_unions++;
    }
    return n_unions;
}

__global__ void boruvka_mst_helper(const int n_vertices,
                                   const int n_edges,
                                   const Edge* edgelist,
                                   Vertex* vertices,
                                   MST* mst) {
    int threadID = threadIdx.y * blockDim.x + threadIdx.x;

    if (threadID == 0) {

        // initialize components
        for (int i = 0; i < n_vertices; i++) {
            vertices[i] = Vertex{i, nullptr};
        }
    }

    int n_components = n_vertices;
    int diff;

    do {
        assign_cheapest(vertices, edgelist, n_edges, threadID);

        if (threadID == 0) {
            diff = update_mst(vertices, edgelist, mst, n_vertices, n_edges);
            n_components -= diff;
        }
    } while (mst->size < n_vertices - 1);
}

MST boruvka_mst(const int n_vertices, const int n_edges, const Edge* edgelist) {
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for Parallel Boruvka's\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("NVIDIA GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }

    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    MST mst;
    mst.size = 0;
    mst.capacity = n_vertices-1;
    mst.weight = 0;
    Edge* mst_tree = (Edge*) malloc(sizeof(Edge) * (n_vertices-1));

    MST* device_mst;
    Edge* device_mst_tree;
    Vertex* device_vertices;
    Edge* device_edgelist;

    cudaMalloc(&device_mst, sizeof(MST));
    cudaMalloc(&device_mst_tree, sizeof(Edge) * (n_vertices-1));
    cudaMalloc(&device_vertices, sizeof(Vertex) * n_vertices);
    cudaMalloc(&device_edgelist, sizeof(Edge) * n_edges);

    cudaMemcpy(device_edgelist, edgelist, sizeof(Edge) * n_edges, cudaMemcpyHostToDevice);
    mst.mst = device_mst_tree;
    cudaMemcpy(device_mst, &mst, sizeof(MST), cudaMemcpyHostToDevice);

    boruvka_mst_helper<<<1, BLOCKSIZE>>>(n_vertices, n_edges, device_edgelist, device_vertices, device_mst);

    cudaMemcpy(&mst, device_mst, sizeof(MST), cudaMemcpyDeviceToHost);
    cudaMemcpy(mst_tree, device_mst_tree, sizeof(Edge) * mst.size, cudaMemcpyDeviceToHost);
    mst.mst = mst_tree;

    cudaFree(device_mst);
    cudaFree(device_mst_tree);
    cudaFree(device_vertices);
    cudaFree(device_edgelist);

    // TODO: Move this into the kernel
    for (int i = 0; i < mst.size; i++) {
        const Edge& e = mst.mst[i];
        mst.weight += e.weight;
    }

    return mst;
}
