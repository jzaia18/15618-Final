# Parallelizing Minimum Spanning Tree Using Borůvka’s Algorithm in Parlaylib and CUDA 
by Shubham Bhargava & Jake Zaia

## Links
* [Project Source Code](https://github.com/jzaia18/15618-Final)
* [Proposal Document](./proposal.pdf)
* [Milestone Report](./milestone.pdf)
* [Final Writeup](./final_writeup.pdf)

## Summary
We implemented Boruvka’s algorithm in [ParlayLib](https://github.com/cmuparlay/parlaylib) and [CUDA](https://developer.nvidia.com/cuda-zone) achieving average computation speedups of 13.31x and 29.05x on the GHC machines using 8 CPU cores and the GeForce RTX 2080 respectively.
ParlayLib is a C++ library written by CMU professors to implement algorithms on multi-core machines.
Boruvka’s algorithm is used to find a Minimum Spanning Forest (MSF) and is more amenable to parallelization than Kruskal’s or Prim’s algorithm.
We tested the algorithm on a variety of sparse and dense graphs with good performance on both the CPU and GPU.
The implementations were benchmarked for performance against a sequential C++ implementation of the algorithm, and tested for correctness against a Python implementation of Kruskal’s algorithm.
The sequential implementation was lightly optimized for running on a single core but was not extensively optimized.
We also wrote graph generators and a benchmarking framework to support testing the correctness and efficiency of our algorithm. 


## Background
Finding the Minimum Spanning Tree (MST) is a common graph problem with many applications in approximate algorithms, network design, image segmentation, and taxonomy.
Many sequential algorithms have been designed for it such as Prim, Kruskal, and Boruvka that run in O(m log n) time.
Boruvka's algorithm is commonly used for parallel applications as it is easy to parallelize in polylogarithmic time.

Boruvka’s algorithm works by using the lightest edge property where the lightest neighboring edge of a vertex is placed in the MST.
By adding the minimum edge for each vertex to the list of MST edges, we get a bunch of connected components.
We can treat these connected components as vertices (removing any self-edges).
Then we repeat the previous steps until all vertices are connected.
This is inherently parallel as the minimum edge for each vertex can be calculated in parallel.

## The Challenge
Implementing Boruvka’s algorithm is tricky as it requires maintaining a shared disjoint-set data structure.
We have to ensure fast memory access to the disjoint-set while ensuring correctness.
Optimizing the disjoint-set might involve figuring out ways to break it down into smaller sets that don’t interact with each other as much.
What makes this truly challenging is that warps in a GPU are SIMD and traversing the disjoint set data structure could result in divergent execution.
Additionally, the disjoint set keeps track of the connected components that act as vertices in each iteration in the algorithm.
This means that we need to contract newly connected components into vertices.
It will be a challenge to choose the most efficient contraction method for this step removing any self-edges.

We are hoping to become better CUDA programmers through this project.
Additionally, we are hoping to apply our knowledge of parallel algorithms beyond just the asymptotics taught in class. 
We are hoping to implement efficient shared data structures suited to the specific problem at hand.

## Platform Choice
We will be coding in C++ so we can use [Parlaylib](https://github.com/cmuparlay/parlaylib) for our CPU parallel implementation.
We will be testing this implementation on the GHC and, if possible, PSC machines to get CPU baselines to compare with our CUDA implementation.
The CUDA implementation will be tested on the lab machines.
The baseline sequential implementation will be done in C++ for fairness.

## Resources
Since we will be implementing this algorithm in CUDA, we will need access to Nvidia GPUs.
We are planning on using the GHC machines with the same setup as assignment 2.
Parlaylib is a header-only library, and thus requires no extra resources.
We will not be using any starter code, however as Boruvka’s is a well-studied algorithm, plenty of pseudocode is available for reference (such as in the 15-210 and 15-852 notes).
It would be interesting to test this project on the PSC machines as well, if possible.

There are a couple of papers, (such as [Paper 1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7092783&tag=1), [Paper 2](https://arxiv.org/pdf/2302.12199.pdf)), that attempt to optimize parallel Boruvka’s that we can use as a reference and possibly compare against.
We can also use the same datasets as the papers used for testing in addition to randomly generated test sets.

## Goals & Deliverables

### Goals
We are aiming to complete at least 3 of the following 4 goals:
- [x] Implement an optimized sequential version of MST using Baruvka’s algorithm for benchmarking the parallel version against.
- [x] Implement a CPU parallel version of Baruvka’s in parlaylib that can be used for further benchmarking.
- [ ] Implement a parallel version of MST in CUDA that scales near-linearly as more GPU threads are used.
- [ ] Implement at least 2 meaningfully different versions of a concurrent disjoint-set. Gather significant benchmark data for a parallel implementation of Boruvka’s running using these different implementations.

### Additional/Stretch Goals
- [ ] Implement a parallel version of MST that performs close to, or better than, the existing implementations in Open MP and MPI.
- [ ] Implement an MST-approximation algorithm that performs better than existing MST implementations while getting close to optimal results.

## Current Status

### 2024-04-16

Over the last few weeks we have implemented Boruvka’s algorithm 3 separate times: sequentially in C++, using parlaylib, and using CUDA. 
The sequential version is lightly optimized for sequential execution and will be used to benchmark our parallel implementations against. 
For our CPU implementation, we have first translated our implementation to Python (in a functional style). 
Then, we converted it to parlaylib code ensuring that our code still matches the theoretical guarantees of the algorithm while maximizing usage of parlaylib primitives as they’ve been optimized for memory usage and scheduling. 
This step involved designing a simple concurrent union-find where we relied on C++ atomics.
Compare and exchange was also useful in updating shared memory when finding the lightest edge per vertex (the first part of the algorithm). 
For the GPU parallel implementation, the CUDA code runs and is correct, but is currently not doing any useful parallelization or obtaining any speedup. 
Optimizing the CUDA implementation will be the focus of the remainder of our effort on this project. 

We have also built up infrastructure for testing our implementations, including a graph generation python script, several datasets, and a python implementation of Kruskal’s algorithm to verify correctness of outputs. 
While these scripts are not the primary portion of our project, they serve an important role in terms of gathering data to benchmark and analyze our code.

## Challenges Faced

Now that we have working implementations for Boruvka’s algorithm, there are some potential hurdles with parallelizing the algorithm which may be difficult to overcome. 
Namely, there are several conditioned loops within the algorithm that terminate early or skip iterations.
These will not perform well for SIMD execution, and we may not be able to circumvent this issue. 
Thus, for the CUDA implementation, it may be the case that a speedup will be miniscule due to warp stall. 
We will have to expend considerable effort to mitigate this issue, and may need to pivot our project to instead focus on benchmarking different implementations for parallel disjoint-set and how they affect the runtime of our implementations. 
This will still produce interesting results since, when trying to parallelize using parlaylib, we realized that there were a lot of different ways to implement the algorithm which made the space of potential optimizations large.

Another challenge has been sourcing data. We expected that it would be easy to get datasets of weighted undirected graphs in similar formats and with various sizes. 
It turns out that accumulating datasets has been a challenge. 
We have created a Python script for generating graphs to use as test data in addition to attempting to use open-source data. 
Obtaining good, diverse, and large datasets remains an ongoing challenge.


## Schedule

Ideal Schedule:

| **Week**     | **Date**     | **Goal**  |
|--------------|--------------|-----------|
| 1            | Apr 1        | ~~Implement sequential version of Baruvka’s algorithm, begin programming CPU-parallel version of Baruvka’s using parlaylib~~ |
| 2            | Apr 8        | ~~Finish implementation of Baruvka’s using parlaylib~~ |
| 3            | Apr 15\*     | ~~Intermediate milestone deadline (Apr 16). Implement Baruvka’s in CUDA: divide parallelizable code into kernels and ensure correctness (without speedup)~~ |
| 4            | Apr 19       | Obtain (any) speedup for finding cheapest edge in CUDA - Jake |
|              | Apr 22       | Implement concurrent disjoint-set data structure in CUDA (ensuring correctness) - Shubham |
| 5            | Apr 25       | Obtain (any) speedup for graph contraction - Jake |
|              | Apr 29       | Have working CUDA code with noticeable speedup. Begin dataset collection - Shubham |
| 6            | May 1        | Fine tune optimizations on CUDA code (remove redundant vertex/edge computation), finalize dataset collection - Jake  |
|              | May 5        | Gather all metrics and benchmark data on GHC (stretch: PSC as well) - Shubham |
|              | May 6\*      | Final poster presentation |

Backup Schedule (in case of goal pivot, which happens after week 4):

| **Week**     | **Date**     | **Goal**  |
|--------------|--------------|-----------|
| 5            | Apr 25       | Implement alternative (wait-free) disjoint-set in parlaylib/open MP - Jake |
|              | Apr 29       | Optimize parlaylib implementation to take advantage of laziness - Shubham |
| 6            | May 1        | Fine tune optimizations on parlaylib code (remove redundant vertex/edge computation), finalize dataset collection - Jake |
|              | May 5        | Gather all metrics and benchmark data on GHC & PSC - Shubham |
|              | May 6\*      | Final poster presentation |
