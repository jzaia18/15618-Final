#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_functions.h>
#include <limits.h>
#include <stdio.h>

#include <chrono>
#include <string>

#include "boruvkas.h"

// TODO: Is it even helpful to think of this as a block for this problem?
// Define constants for CUDA threadblocks
#define BLOCKSIZE 64
#define BLOCKCOUNT 64
#define THREADCOUNT (BLOCKSIZE * BLOCKCOUNT)

#define NO_EDGE (INT_MAX)

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {
    Vertex* vertices;
    Edge* edges;
    int* mst_tree;
    int n_vertices;
    int n_edges;
};

// Another global value
__device__ int n_components;

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstGraphParams;

__device__ inline int get_component(Vertex* componentlist, const int i) {
    int curr = componentlist[i].component;

    while (componentlist[curr].component != curr) {
        curr = componentlist[curr].component;
    }

    return curr;
}

__device__ inline void merge_components(Vertex* componentlist, const int i,
                                        const int j) {
    int u = i;
    int v = j;
    int old;
    do {
        u = get_component(componentlist, u);
        old = atomicCAS(&(componentlist[u].component), u, v);
    } while (old != u);
}

__global__ void init_arrs() {
    const int threadID = threadIdx.x;

    const int start = (threadID * cuConstGraphParams.n_vertices) / BLOCKSIZE;
    const int end =
        ((threadID + 1) * cuConstGraphParams.n_vertices) / BLOCKSIZE;

    // initialize components
    for (int i = start; i < end; i++) {
        cuConstGraphParams.vertices[i] = Vertex{i, NO_EDGE};
    }
}

__global__ void reset_arrs() {
    const int threadID = threadIdx.x;

    const int start = (threadID * cuConstGraphParams.n_vertices) / BLOCKSIZE;
    const int end =
        ((threadID + 1) * cuConstGraphParams.n_vertices) / BLOCKSIZE;

    // initialize components
    for (int i = start; i < end; i++) {
        cuConstGraphParams.vertices[i].cheapest_edge = NO_EDGE;
    }
}

__global__ void assign_cheapest() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    const int start = (threadID * cuConstGraphParams.n_edges) / THREADCOUNT;
    const int end = ((threadID + 1) * cuConstGraphParams.n_edges) / THREADCOUNT;

    Vertex* vertices = cuConstGraphParams.vertices;

    for (int i = start; i < end; i++) {
        Edge& e = cuConstGraphParams.edges[i];
        e.u = get_component(vertices, e.u);
        e.v = get_component(vertices, e.v);

        // Skip edges that connect a component to itself
        if (e.u == e.v) {
            continue;
        }

        // Atomic update cheapest_edge[u]
        int old = vertices[e.u].cheapest_edge;
        while (old == NO_EDGE ||
               e.weight < cuConstGraphParams.edges[old].weight) {
            old = atomicCAS(&vertices[e.u].cheapest_edge, old, i);
        }

        // Atomic update cheapest_edge[v]
        old = vertices[e.v].cheapest_edge;
        while (old == NO_EDGE ||
               e.weight < cuConstGraphParams.edges[old].weight) {
            old = atomicCAS(&vertices[e.v].cheapest_edge, old, i);
        }
    }
}

__global__ void update_mst() {
    const int threadID = threadIdx.x;

    const int start = (threadID * cuConstGraphParams.n_vertices) / BLOCKSIZE;
    const int end =
        ((threadID + 1) * cuConstGraphParams.n_vertices) / BLOCKSIZE;

    int n_unions = 0;
    // Connect newest edges to MST
    for (int i = start; i < end; i++) {
        const int edge_ind = cuConstGraphParams.vertices[i].cheapest_edge;

        if (edge_ind == NO_EDGE) {
            continue;
        }

        const Edge& edge_ptr = cuConstGraphParams.edges[edge_ind];

        // If this edge is covered twice, only union when i == u (u < v is
        // assumed)
        if (edge_ptr.v == i &&
            edge_ind == cuConstGraphParams.vertices[edge_ptr.u].cheapest_edge) {
            continue;
        }

        cuConstGraphParams.mst_tree[edge_ind] = 1;
        merge_components(cuConstGraphParams.vertices, edge_ptr.u, edge_ptr.v);
        n_unions++;
    }

    // TODO better method available?
    atomicSub(&n_components, n_unions);
}

MST boruvka_mst(const int n_vertices, const int n_edges, const Edge* edgelist) {
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for Parallel Boruvka's\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("NVIDIA GeForce RTX 2080") == 0) {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }

    printf("---------------------------------------------------------\n");
    if (!isFastGPU) {
        printf(
            "WARNING: "
            "You're not running on a fast GPU, please consider using "
            "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    int* device_mst_tree;
    Vertex* device_vertices;
    Edge* device_edgelist;

    cudaMalloc(&device_mst_tree, sizeof(int) * n_edges);
    cudaMemset(device_mst_tree, 0, sizeof(int) * n_edges);

    cudaMalloc(&device_vertices, sizeof(Vertex) * n_vertices);

    cudaMalloc(&device_edgelist, sizeof(Edge) * n_edges);
    cudaMemcpy(device_edgelist, edgelist, sizeof(Edge) * n_edges,
               cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.vertices = device_vertices;
    params.edges = device_edgelist;
    params.mst_tree = device_mst_tree;
    params.n_edges = n_edges;
    params.n_vertices = n_vertices;

    cudaMemcpyToSymbol(cuConstGraphParams, &params, sizeof(GlobalConstants));

    // Run Boruvka's in parallel
    int n_comp = n_vertices;
    int n_comp_old;

    // Initialise global
    cudaMemcpyToSymbol(n_components, &n_comp, sizeof(int));

    init_arrs<<<1, BLOCKSIZE>>>();

    do {
        n_comp_old = n_comp;

        reset_arrs<<<1, BLOCKSIZE>>>();
        assign_cheapest<<<BLOCKCOUNT, BLOCKSIZE>>>();
        update_mst<<<1, BLOCKSIZE>>>();
        cudaMemcpyFromSymbol(&n_comp, n_components, sizeof(int));

        // debug
        // Vertex * ts = (Vertex *) malloc(sizeof(Vertex) * n_vertices);
        // cudaMemcpy(ts, device_vertices, sizeof(Vertex) * n_vertices,
        // cudaMemcpyDeviceToHost); for (int i = 0; i < n_vertices; i++) {
        //     printf("%d ", ts[i].cheapest_edge);
        // }
        // printf("\n");
        // Edge * ed = (Edge *) malloc(sizeof(Edge) * n_vertices);
        // cudaMemcpy(ed, device_edgelist, sizeof(Edge) * n_edges,
        // cudaMemcpyDeviceToHost); for (int i = 0; i < n_edges; i++) {
        //     printf("%d-%d-%d ", ed[i].u, ed[i].v, ed[i].weight);
        // }
        // printf("nc %d\n", n_comp);
    } while (n_comp != n_comp_old && n_comp > 1);

    MST mst;
    mst.weight = 0;
    int* mst_tree = (int*)malloc(sizeof(int) * n_edges);

    // Copy run results off of device
    cudaMemcpy(mst_tree, device_mst_tree, sizeof(int) * n_edges,
               cudaMemcpyDeviceToHost);
    mst.mst = mst_tree;

    // Clean up device memory
    cudaFree(device_mst_tree);
    cudaFree(device_vertices);
    cudaFree(device_edgelist);

    // TODO: Move this into the kernel (filtering vertices to get a short list)
    // Compute final weight
    for (int i = 0; i < n_edges; i++) {
        if (mst.mst[i] == 1) {
            const Edge& e = edgelist[i];
            mst.weight += e.weight;
        }
    }

    return mst;
}
