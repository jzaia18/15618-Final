#include <chrono>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include "boruvkas.h"

// Define constants for CUDA threadblocks
#define THREADBLOCK_WIDTH (8)
#define THREADBLOCK_HEIGHT (8)
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


__global__ void test(int *a) {
    int threadID = threadIdx.y * blockDim.x + threadIdx.x;
    a[threadID] = threadID;
}

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

std::vector<Edge>* boruvka_mst_helper(int n_vertices, const std::vector<Edge>& edgelist) {
    std::vector<Edge>* mst = new std::vector<Edge>();
    std::vector<Vertex> vertices(n_vertices);

    // initialize components
    for (int i = 0; i < n_vertices; i++) {
        vertices[i] = Vertex{i, nullptr}; //Vertex{i, i, nullptr};
    }

    int n_components = n_vertices;
    bool keep_going;

    do {
        keep_going = false;
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

        keep_going = false;
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
            keep_going = true;
        }

    } while (keep_going && n_components > 1);

    return mst;
}

MST boruvka_mst(int n_vertices, const std::vector<Edge>& edgelist) {
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

    // TODO: all this code is silly, just tests that we have CUDA set up correctly
    {
        int a[BLOCKSIZE] = {0};
        int *d_a;

        // Allocate device memory for a
        cudaMalloc((void**)&d_a, sizeof(int) * BLOCKSIZE);


        for (int i = 0; i < BLOCKSIZE; i++) {
            a[i] = 0;
        }
        for (int i = 0; i < BLOCKSIZE; i++) {
            printf("%d ", a[i]);
        }
        printf("\n");


        // Transfer data from host to device memory
        cudaMemcpy(d_a, a, sizeof(int) * BLOCKSIZE, cudaMemcpyHostToDevice);

        test<<<1, BLOCKSIZE>>>(d_a);

        // Transfer data from device to host memory
        cudaMemcpy(a, d_a, sizeof(int) * BLOCKSIZE, cudaMemcpyDeviceToHost);

        for (int i = 0; i < BLOCKSIZE; i++) {
            printf("%d ", a[i]);
        }
        printf("\n");

        cudaFree(d_a);
    }

    MST result;
    result.mst = boruvka_mst_helper(n_vertices, edgelist);

    result.weight = 0;
    for (const Edge& e : *result.mst) {
        result.weight += e.weight;
    }

    return result;
}
