#include <chrono>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <limits.h>

#include "boruvkas.h"

// TODO: Is it even helpful to think of this as a block for this problem?
// Define constants for CUDA threadblocks
#define THREADBLOCK_WIDTH (8)
#define THREADBLOCK_HEIGHT (8)
#define BLOCKSIZE (THREADBLOCK_WIDTH*THREADBLOCK_HEIGHT)

#define NO_EDGE (INT_MAX)

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {
    // TODO
};

// Another global value
__device__ int n_components;

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;


__device__ inline int get_component(Vertex* componentlist, const int i) {
    int curr = componentlist[i].component;

    while (componentlist[curr].component != curr) {
        curr = componentlist[curr].component;
    }

    // Flatten component trees
    if (componentlist[i].component != curr) {
        atomicExch(&componentlist[i].component, curr);
    }
    return curr;
}

__device__ inline void merge_components(Vertex* componentlist, const int i, const int j) {
    componentlist[get_component(componentlist, i)].component = j;
    // const int ci = get_component(componentlist, i);
    // const int cj = get_component(componentlist, j);
    // componentlist[ci].component = cj;
}

__global__ void assign_cheapest(Vertex* vertices,
                                const Edge* edgelist,
                                const int n_edges) {
    const int threadID = threadIdx.y * blockDim.x + threadIdx.x;

    const int start = (threadID * n_edges / BLOCKSIZE);
    const int end = ((threadID+1) * n_edges / BLOCKSIZE);

    for (int i = start; i < end; i++) {
        const Edge& e = edgelist[i];
        const int c1 = get_component(vertices, e.u);
        const int c2 = get_component(vertices, e.v);

        // Skip edges that connect a component to itself
        if (c1 == c2) {
            continue;
        }

        if (i < vertices[c1].cheapest_edge) {
            atomicMin(&vertices[c1].cheapest_edge, i);
        }
        if (i < vertices[c2].cheapest_edge) {
            atomicMin(&vertices[c2].cheapest_edge, i);
        }

        // atomicMin(&vertices[c1].cheapest_edge, i);
        // atomicMin(&vertices[c2].cheapest_edge, i);

        // if (vertices[c1].cheapest_edge == NO_EDGE || edge_cmp(e, edgelist[vertices[c1].cheapest_edge]) < 0) {
        //     vertices[c1].cheapest_edge = i;
        // }
        // if (vertices[c2].cheapest_edge == NO_EDGE || edge_cmp(e, edgelist[vertices[c2].cheapest_edge]) < 0) {
        //     vertices[c2].cheapest_edge = i;
        // }
    }
}


__global__ void update_mst(Vertex* vertices,
                                 const Edge* edgelist,
                                 MST* mst,
                                 const int n_vertices,
                                 const int n_edges) {
    const int threadID = threadIdx.y * blockDim.x + threadIdx.x;

    if (threadID != 0) return;

    int n_unions = 0;
    // Connect newest edges to MST
    for (int i = 0; i < n_vertices; i++) {
        const int edge_ind = vertices[i].cheapest_edge;

        if (edge_ind == NO_EDGE) {
            continue;
        }

        const Edge& edge_ptr = edgelist[edge_ind];

        // if (get_component(vertices, edge_ptr.u) == get_component(vertices, edge_ptr.v)) {
        //     continue;
        // }

        mst->mst[mst->size++] = edge_ind;
        vertices[get_component(vertices, edge_ptr.u)].cheapest_edge = NO_EDGE;
        vertices[get_component(vertices, edge_ptr.v)].cheapest_edge = NO_EDGE;
        merge_components(vertices, edge_ptr.u, edge_ptr.v);
        n_unions++;
    }
    atomicSub(&n_components, n_unions);
}

// __global__ void boruvka_mst_helper(const int n_vertices,
//                                    const int n_edges,
//                                    const Edge* edgelist,
//                                    Vertex* vertices,
//                                    MST* mst) {
//     const int threadID = threadIdx.y * blockDim.x + threadIdx.x;

//     __shared__ int diff;

//     do {
//         assign_cheapest(vertices, edgelist, n_edges, threadID);

//         __syncthreads();

//         if (threadID == 0) {
//             diff = update_mst(vertices, edgelist, mst, n_vertices, n_edges);
//         }

//         __syncthreads();
//     } while (diff != 0);
// }

__global__ void init_arrs(const int n_vertices,
                          Vertex* vertices) {
    const int threadID = threadIdx.y * blockDim.x + threadIdx.x;

    const int start = (threadID * n_vertices / BLOCKSIZE);
    const int end = ((threadID+1) * n_vertices / BLOCKSIZE);

    // initialize components
    for (int i = start; i < end; i++) {
        vertices[i] = Vertex{i, NO_EDGE};
    }

    if (threadID == 0) {
        n_components = n_vertices;
    }
}

void initGPUs() {
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
}

MST boruvka_mst(const int n_vertices, const int n_edges, const Edge* edgelist) {
    MST mst;
    mst.size = 0;
    mst.capacity = n_vertices-1;
    mst.weight = 0;
    int* mst_tree = (int*) malloc(sizeof(int) * (n_vertices-1));

    MST* device_mst;
    int* device_mst_tree;
    Vertex* device_vertices;
    Edge* device_edgelist;

    cudaMalloc(&device_mst, sizeof(MST));
    cudaMalloc(&device_mst_tree, sizeof(int) * (n_vertices-1));
    cudaMalloc(&device_vertices, sizeof(Vertex) * n_vertices);
    cudaMalloc(&device_edgelist, sizeof(Edge) * n_edges);

    cudaMemcpy(device_edgelist, edgelist, sizeof(Edge) * n_edges, cudaMemcpyHostToDevice);
    mst.mst = device_mst_tree;
    cudaMemcpy(device_mst, &mst, sizeof(MST), cudaMemcpyHostToDevice);

    // Run Boruvka's in parallel
    init_arrs<<<1, BLOCKSIZE>>>(n_vertices, device_vertices);
    int n_comp = n_vertices;
    int n_comp_old;

    do {
        n_comp_old = n_comp;
        assign_cheapest<<<1, BLOCKSIZE>>>(device_vertices, device_edgelist, n_edges);

        update_mst<<<1, BLOCKSIZE>>>(device_vertices, device_edgelist, device_mst, n_vertices, n_edges);
        cudaMemcpyFromSymbol(&n_comp, n_components, sizeof(int));
    } while (n_comp != n_comp_old && n_comp > 1);

    // Copy run results off of device
    cudaMemcpy(&mst, device_mst, sizeof(MST), cudaMemcpyDeviceToHost);
    cudaMemcpy(mst_tree, device_mst_tree, sizeof(int) * mst.size, cudaMemcpyDeviceToHost);
    mst.mst = mst_tree;

    // Clean up device memory
    cudaFree(device_mst);
    cudaFree(device_mst_tree);
    cudaFree(device_vertices);
    cudaFree(device_edgelist);

    // TODO: Move this into the kernel that performs a scan
    // Compute final weight
    for (int i = 0; i < mst.size; i++) {
        const Edge& e = edgelist[mst.mst[i]];
        mst.weight += e.weight;
    }

    return mst;
}
