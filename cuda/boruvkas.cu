#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_functions.h>
#include <limits.h>
#include <stdio.h>

#include <chrono>
#include <string>

#include "boruvkas.h"

// Define constants for CUDA threadblocks
#define BLOCKSIZE (1024)

#define NBLOCKS_ASSIGN_CHEAPEST (128)
#define NBLOCKS_OTHER (128)

#define NTHREADS_ASSIGN_CHEAPEST (NBLOCKS_ASSIGN_CHEAPEST * BLOCKSIZE)
#define NTHREADS_OTHER (NBLOCKS_OTHER * BLOCKSIZE)

#define NO_EDGE (ULONG_MAX)

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {
    Vertex* vertices;
    Edge* edges;
    char* mst_tree;
    ullong n_vertices;
    ullong n_edges;
};

// Another global value
__device__ ullong n_unions_total;

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstGraphParams;

__device__ inline int edge_cmp(const Edge* edges, const ullong i, const ullong j)
{
    if (i == j) return 0;

    const Edge& lhs = edges[i];
    const Edge& rhs = edges[j];

    if (lhs.weight < rhs.weight) {
        return -1;
    }
    if (lhs.weight > rhs.weight) {
        return 1;
    }

    if (i < j) {
        return -1;
    }
    return 1;
}

__device__ inline ullong get_component(Vertex* componentlist, const ullong i) {
    ullong curr = componentlist[i].component;

    // while (componentlist[curr].component != curr) {
    //     curr = componentlist[curr].component;
    // }

    return curr;
}

__device__ inline void flatten_component(Vertex* componentlist, const ullong i) {
    ullong curr = componentlist[i].component;

    while (componentlist[curr].component != curr) {
        curr = componentlist[curr].component;
    }

    // Flatten component trees
    componentlist[i].component = curr;
    // if (componentlist[i].component != curr) {
    //     atomicExch(&componentlist[i].component, curr);
    // }
}


__device__ inline void merge_components(Vertex* componentlist, const ullong i,
                                        const ullong j) {
    // ullong u = i;
    // ullong v = j;
    componentlist[i].component = j;
    // const ullong v = get_component(componentlist, j);
    // ullong old;
    // do {
    //     u = get_component(componentlist, u);
    //     old = atomicCAS(&(componentlist[u].component), u, v);
    // } while (old != u);
}

__global__ void init_arrs() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    const ullong n_vertices = cuConstGraphParams.n_vertices;
    Vertex* const vertices = cuConstGraphParams.vertices;

    const ullong start = (threadID * n_vertices / NTHREADS_OTHER);
    const ullong end = ((threadID + 1) * n_vertices / NTHREADS_OTHER);

    // initialize components
    for (ullong i = start; i < end; i++) {
        vertices[i] = Vertex{i, NO_EDGE};
    }
}

__global__ void reset_arrs() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    const ullong n_vertices = cuConstGraphParams.n_vertices;
    Vertex* const vertices = cuConstGraphParams.vertices;

    const ullong start = (threadID * n_vertices / NTHREADS_OTHER);
    const ullong end = ((threadID + 1) * n_vertices / NTHREADS_OTHER);

    // initialize components
    for (ullong i = start; i < end; i++) {
        vertices[i].cheapest_edge = NO_EDGE;
        flatten_component(vertices, i);
    }
}

__global__ void assign_cheapest() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    // Renaming to make life easier, this gets compiled away
    const ullong n_edges = cuConstGraphParams.n_edges;
    Vertex* const vertices = cuConstGraphParams.vertices;
    Edge* const edges = cuConstGraphParams.edges;

    const ullong start = (threadID * n_edges) / NTHREADS_ASSIGN_CHEAPEST;
    const ullong end = ((threadID + 1) * n_edges) / NTHREADS_ASSIGN_CHEAPEST;

    for (ullong i = start; i < end; i++) {
        Edge& e = edges[i];
        e.u = get_component(vertices, e.u);
        e.v = get_component(vertices, e.v);

        // Skip edges that connect a component to itself
        if (e.u == e.v) {
            continue;
        }

        // Atomic update cheapest_edge[u]
        ullong expected = vertices[e.u].cheapest_edge;
        ullong old;
        while (expected == NO_EDGE || edge_cmp(edges, i, expected) < 0) {
            old = atomicCAS(&vertices[e.u].cheapest_edge, expected, i);
            if (expected == old) {
                break;
            }
            expected = old;
        }

        // Atomic update cheapest_edge[v]
        expected = vertices[e.v].cheapest_edge;
        while (expected == NO_EDGE || edge_cmp(edges, i, expected) < 0) {
            old = atomicCAS(&vertices[e.v].cheapest_edge, expected, i);
            if (expected == old) {
                break;
            }
            expected = old;
        }
    }
}

__global__ void update_mst() {
    const int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    // Renaming to make life easier, this gets compiled away
    const ullong n_vertices = cuConstGraphParams.n_vertices;
    Vertex* const vertices = cuConstGraphParams.vertices;
    Edge* const edges = cuConstGraphParams.edges;

    const ullong start = (threadID * n_vertices) / NTHREADS_OTHER;
    const ullong end = ((threadID + 1) * n_vertices) / NTHREADS_OTHER;

    ullong n_unions_made = 0;
    // Connect newest edges to MST
    for (ullong i = start; i < end; i++) {
        const ullong edge_ind = vertices[i].cheapest_edge;

        if (edge_ind == NO_EDGE) {
            continue;
        }

        const Edge& edge_ptr = edges[edge_ind];

        // If this edge is covered twice, only union when i == u (u < v is
        // assumed)
        if (edge_ptr.v == i &&
            edge_ind == vertices[edge_ptr.u].cheapest_edge) {
            continue;
        }

        const ullong j = (i == edge_ptr.u? edge_ptr.v : edge_ptr.u); // this is the other index

        cuConstGraphParams.mst_tree[edge_ind] = 1;
        merge_components(vertices, i, j);
        n_unions_made++;
    }

    atomicAdd(&n_unions_total, n_unions_made);
}

void initGPUs() {
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
}

MST boruvka_mst(const ullong n_vertices, const ullong n_edges, const Edge* edgelist) {
    MST mst;
    mst.weight = 0;
    char* mst_tree = (char*) malloc(sizeof(char) * n_edges);

    char* device_mst_tree;
    Vertex* device_vertices;
    Edge* device_edgelist;

    cudaMalloc(&device_mst_tree, sizeof(char) * n_edges);
    cudaMemset(device_mst_tree, 0, sizeof(char) * n_edges);

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
    ullong n_unions = 0;
    ullong n_unions_old;

    // Initialise global
    cudaMemcpyToSymbol(n_unions_total, &n_unions, sizeof(ullong));

    init_arrs<<<NBLOCKS_OTHER, BLOCKSIZE>>>();

    do {
        n_unions_old = n_unions;

        reset_arrs<<<NBLOCKS_OTHER, BLOCKSIZE>>>();
        assign_cheapest<<<NBLOCKS_ASSIGN_CHEAPEST, BLOCKSIZE>>>();
        update_mst<<<NBLOCKS_OTHER, BLOCKSIZE>>>();
        cudaMemcpyFromSymbol(&n_unions, n_unions_total, sizeof(ullong));

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
    } while (n_unions != n_unions_old && n_unions < n_vertices - 1);

    // Copy run results off of device
    cudaMemcpy(mst_tree, device_mst_tree, sizeof(char) * n_edges,
               cudaMemcpyDeviceToHost);
    mst.mst = mst_tree;

    // Clean up device memory
    cudaFree(device_mst_tree);
    cudaFree(device_vertices);
    cudaFree(device_edgelist);

    // TODO: Move this into the kernel (filtering vertices to get a short list)
    // Compute final weight
    for (ullong i = 0; i < n_edges; i++) {
        if (mst.mst[i]) {
            const Edge& e = edgelist[i];
            mst.weight += e.weight;
        }
    }

    return mst;
}
