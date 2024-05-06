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

## Approach
To represent the graph, we chose to use a list of edges as opposed to an adjacency list as it becomes very easy to parallelize the first step.
As we loop through the edge list, we can check if that edge is the cheapest edge for either of its neighboring vertices.
We need to store the cheapest edge per vertex.
This would require synchronization to ensure multiple cores do not attempt to update the cheapest edge per vertex simultaneously.
If we used an adjacency list, we would have to parallelize over the vertices, but if the vertices had vastly different numbers of edges, this could result in bad load balancing.
However, if each core was assigned one vertex, we wouldn’t need to deal with synchronization in this step.
This approach might have had slightly better locality in updating the cheapest edge array, however, we address this issue with our edge list in a later section.
The edge list representation also means that we cannot start the second part of the algorithm until we have gone through all the edges.
However, due to the relatively even workload among cores/threads, this turns out to be a non-issue.

We also have to be mindful of breaking ties between edges with the same weight at this step, a cause of much pain when debugging.
If edges of the same weight form a cycle, it is possible that all these edges are chosen simultaneously as the cheapest edge per vertex and then added to the MST.
Cycles are disallowed for two reasons, the first being that the final result is no longer a valid tree/forest, and the second being that it can cause the program to hang when traversing a part of the tree.
As such, we need some way of strictly ordering all the edges (that is consistent with ordering them by weight).
If two edges have the same weight, we choose the edge with the lower index in the edge list.
This will ensure that we can’t select a cycle of edges with the same weight to be a part of the MST in one round of the algorithm.
Initially, we used a different ordering which turned out to be buggy for subtle reasons discussed later. 

Before we do the graph contraction, we must also add the edges to our MST.
This was combined with the second step in our implementations.
In ParlayLib, we used the parlay::sequence primitive to store the list of edges in the MST.
Every round, we would use a filter on our list of cheapest edges to get rid of edges that might be double counted and then append them to our previous list which Parlay handles efficiently.
Since we did not have an easy and efficient filter and append available to us in CUDA, we instead kept a boolean for each of the original edges.
As we iterate through the cheapest edges in parallel, we update the boolean corresponding to the edge.
We had to be careful about not double counting edges as we kept track of the number of edges added to our MST (which was used to determine when to terminate our algorithm in CUDA implementation).
We could check if an edge was already counted by checking our MST boolean array atomically.
Alternatively, we could check that the current edge did not count as the cheapest edge for any other vertex.
In practice, we didn’t notice any significant difference and used the second implementation.

We used a concurrent union-find data structure to keep track of which vertices had been merged into a single vertex.
We call each set of vertices in a contracted portion of the graph a “component”.
A component forms a tree with some root vertex responsible for maintaining information about the component itself.
For our particular implementation, the union-find was stored as an array of length n, where n represents the number of vertices in the entire graph.
Each array element tracks the cheapest edge for the given vertex as well as a parent in the component to which the vertex belongs.
For root nodes of a component, the vertex stores itself.
To union 2 components, the root of one component must set its parent to any node contained in the other.
To find the root of a component, you must loop over the “parent” nodes until the root is reached.
However, in our parallel implementation we carefully structured our usage of union and find operations such that we could be sure that the component trees never had a height greater than 2 (this is further detailed in the optimizations section below).
This allowed us to omit the loop in the find operation entirely, significantly cutting down on the time spent on such operations.
Since find operations make up a significant amount of the implementation of Boruvka’s algorithm, our custom problem-optimized union-find implementation significantly outperformed our CUDA adaptation of the concurrent union-find from Wait-free Parallel Algorithms for the Union-Find Problem, achieving a 1.077x speedup over our implementation of the version from this paper.

The union-find data structure formed a key component of our graph contractions.
Once we combine the vertices, we have to update our edge list to reflect the updated contracted graph.
One way to do this is to update each edge.
We would replace the vertices of an edge with the component the vertex belongs to.
As such the edge would now go between 2 components of our updated graph.
This can be done efficiently using the find operation.
We noticed that this can be combined with the first step of the algorithm: as we iterate through the edges on the algorithm's first step, we could use the union-find to find the component for each edge’s endpoint and update the edge in the edge list.
This worked well for both our ParlayLib and CUDA implementations.
For our CUDA implementation, we also had to worry about warp sync which meant that we would flatten our union-find trees as discussed later.

One might notice that in this case, the number of edges doesn’t go down throughout the algorithm.
Instead, we obtain self-edges and multi-edges when we merge vertices.
Since self-edges can no longer contribute to the spanning tree, it is possible to filter them out. 
For the ParlayLib implementation, we implemented this which means that in every iteration, the number of edges goes down. 
However, due to the nature of the algorithm, for many types of random/unstructured graphs, the number of self-edges is low. 
As such, the number of edges in the edge list goes down very slowly despite the number of vertices nearly halving every iteration. 
The speedup obtained varied across test cases but was overall small. 
In addition, filtering the edge list results in having to track extra auxiliary information about the edges which cancels most of the benefits of filtering out self-edges. 
Implementing this filtering would be tricky in CUDA so we tried using the CUB and Thrust libraries with no success. 
Due to the negligible benefits seen in the Parlay implementation, we decided not to implement it in CUDA.

In addition to filtering self-edges, we could also replace multi-edges with the shortest edge between the two vertices. 
However, this could take up to O(E + n2) space and time where n is the current number of components. 
As such, this is not done in Boruvka’s algorithm. 
However, when the number of components becomes sufficiently small, we can get rid of most of the unnecessary edges in the multi-edge. 
At this small scale, it likely even makes sense to switch from a CUDA implementation of Boruvka’s algorithm to a parallel implementation of the Filter-Kruskal algorithm. 
This kind of optimization requires a good number of changes to our CUDA implementation and implementing Filter-Kruskal was beyond the scope of this project.

In our ParlayLib implementation, some of the steps mentioned earlier were combined or separated depending on the needs. 
By combining steps in Parlay, we can avoid doing multiple passes over our data structures. 
Parlay also provides lazy data structures which allows us to chain certain operations easily and prevents Parlay’s scheduler from doing multiple passes over the data. 
This can partly be seen when we use map_maybe in Parlay which combines a map and a filter. 
Since our ParlayLib implementation uses edge filtering, we terminate our main loop of the algorithm when there are no more edges left across any disjoint components. 
This means that each of our connected components has been merged into a single component and as such, we are done.

For the CUDA implementation, we broke the implementation of our main loop into three kernels. 
These kernels were chosen to combine similar operations and maximize locality. 
The first kernel resets the cheapest edge array as well as flattens our union-find data structure (this sets up the second kernel to perform find on a flat tree). 
Although it would be possible to merge the functionality of this kernel into the other two, it does not perform as well. 
The second kernel, assign_cheapest() corresponds to the first portion of the algorithm which involves finding the cheapest step. 
The third kernel, update_mst() corresponds to contracting the graph. 
This involves iterating through our cheapest edge array in parallel to merge vertices along those edges. 
Finally, the host copies a value from the device to determine when the MST is fully formed, and these 3 kernels are iterated until this point.

### Tie-Breaking
As mentioned before, the method of tie-breaking for edges with the same weight is extremely important.
If the tie-breaking method is incoherent it can cause cycles to form, which presents a correctness issue and can cause the program to hang. 
At first, we used a method of ordering edges based on the vertices that the edge conjoined, u and v, choosing that edges with lower values of u than v would come “first”.
This is a coherent method of breaking ties; it is even one of the examples listed in the Wikipedia article on Boruvka’s algorithm.
However, this fails to be true when accounting for an optimization we make in the code.
When selecting edges we alter the values u and v to be the root of the associated component.
This simplifies comparing edges between components that have many vertices.
However, this also introduces multi-edges, meaning that the original method of comparing edges no longer has a strict ordering for some edges with the same weight.
The result is undefined behavior wherein cycles may be added.
Interestingly, this issue only appears in a parallel implementation, since a sequential implementation will visit all edges in some sequential order, which prevents 2 different but equal-weight edges from being added to 2 components simultaneously, preventing the creation of such cycles.

### Synchronization
Synchronization was tricky to get right for our project, especially since we hand-rolled our own wait-free union-find. 
Our union-find wouldn’t work in a general application but was designed to be specific to our needs. 
In addition to using a union-find, we had to ensure that we correctly found the cheapest edge for every vertex. 
This couldn’t simply be done using an atomic minimum so we settled on using CAS. 
For the CUDA implementation, we were tracking the number of components/vertices in the graph.
This was updated anytime we merged two vertices forcing us to atomically handle this as well.

For ParlayLib, we were initially using the union-find provided by ParlayLib. 
It implemented a variation of path compression (grandparent-linking) for find and provided a simplistic union that only worked in very specific algorithms. 
To address this, we implemented a simple union ourselves. However, it didn’t mesh well with the find implementation provided by ParlayLib. 
Occasionally, our program would hang and we couldn’t understand why. Any issues showed up only at 8 cores for large test cases rarely. 
Eventually, we reimplemented the find function ourselves getting rid of any path compression/shortening. 
We reasoned by producing small counterexamples that the grandparent-linking interfered with our implementation of the union function. 
The cause of the bug was loops being created in our union-find implementation under certain circumstances.
We used C++ atomics for the arrays/sequences that could be simultaneously modified by multiple threads. 
There was a minor implementation issue with how we used compare_exchange_strong due to misunderstanding its unusual semantics. 
We used a compare-and-compare-and-swap as opposed to a compare-and-swap in an attempt to optimize the code (this made no practical difference in speed across tests). 
As a result, any bugs arising from misusing compare_exchange_strong were incredibly rare as the first compare would handle most synchronization issues. 
This made it nearly impossible to debug. 
We attempted to simplify the code by removing the compare before compare-and-swap leading to many more failing test cases which pointed us toward the problem. 

### Optimizations for CUDA

Parallelizing Boruvka’s algorithm using CUDA presents a challenge since warps execute instructions as SIMD. 
Boruvka’s algorithm is composed of many loops that have unpredictable early exit points and branches. 
Thus, a naive translation to CUDA will produce code that has significant thread divergence and warp stall. 
Indeed, our original implementation would spend on average over 1000 cycles per operation stalled in the worst-performing kernels. 
This constituted well over 90% of cycles spent executing the kernel. 
Through careful optimization, we were able to reduce this to approximately 70 cycles in the worst case by carefully organizing the access patterns of edges.
The first optimization is to ensure that (for threads within a threadblock) all threads are accessing similar memory locations. 
We were able to accomplish this by ensuring that all edges are ordered in increasing order of the first vertex. 
This means that sequential edges are very likely to act on at least 1 vertex in common. 
We split the edge list into chunks where each threadblock was given 1 chunk to process. 
Then, we had threads within a threadblock traverse this chunk in interleaved order. 
The result of this is significantly reduced memory stall times, since each edge only occupies 12 bytes and several edges can be read in the same cache line. 
Moreover, while contention on single vertices is high, this contention is mostly contained within a threadblock (and even more often within a warp), which is significantly preferred to distributing the contention across different threadblocks.

An algorithmic optimization that is often made for Boruvka’s algorithm is to flatten the component trees as they are traversed. 
Ordinarily, this can be done as each component is accessed. 
This is typically done by updating the parent pointers as the component tree is traversed. 
However, for SIMD execution this is suboptimal since it means that some threads will spend many more iterations looping than others, which can cause thread divergence. 
Moreover, checking the component associated with a vertex is one of the most common subroutines used in Boruvka’s algorithm, so it must be a quick operation. 
We eliminated the looping behavior by introducing an additional “flatten” step that can be performed in parallel between executions of the main portion of the algorithm. 
In essence, after the MST is updated, each vertex traverses its component tree to its root and stores its root, flattening the tree maximally. 
Then, until the next merge step every single component is guaranteed to be either a root or of depth 1. 
Notably, for this to remain threadsafe, there is one more precaution that must be taken: when merging components, each component must only be merged as the “child” of another component once.
If this is not the case, then merges will overwrite the parent since it cannot loop to retrieve the root component. 
This is different from the typical optimization for a sequential version of a union-find which would instead track a rank or depth of the component trees and merge the smaller as the parent. 

Another optimization we attempted was to keep two copies of every edge in our edge list.
As mentioned before, using an adjacency list to represent the graph would mean that we would experience better locality as we tried to find the cheapest edge per vertex. 
This is because, for each vertex, we can iterate through its neighbors picking the cheapest one. 
Instead, parallelizing over the edges, (u, v, w), we would attempt to update the cheapest edge for u and then v. 
This involved accessing different portions of the cheapest edge array. Note that the edges were already sorted by u which means that updating the cheapest edge for vertex u had temporal locality. 
However, updating the cheapest edge for vertex v didn’t have the same locality. 
For a given edge, (u, v, w), if we stored both (u, v, w) and (v, u, w) in our edge list, we could only update the cheapest edge array for the first vertex in the edge. 
If these edges were all sorted by the first component, we would gain the same locality benefits as an adjacency list. 
However, implementing this involved a lot of tradeoffs. 
To create this sorted edge list efficiently, we had to convert our old edge list to an adjacency list and then back to this new “directed” edge list representation. 
Additionally, we had to store extra auxiliary information for each edge. 
Previously, we would memory-map our large binary graph files directly into memory. 
That was no longer possible due to the changed edge format. We also had to update our edge tie-breaking function. 
Furthermore, our initial communication from the CPU to the GPU increased as we had to send more auxiliary information. 
Although we saw a slight improvement in CUDA running time, our preprocessing time and our total execution time suffered greatly. 
It also made the code more convoluted and as such, we opted not to do this.

### Project Infrastructure
A significant portion of time was spent producing infrastructure to back this project. 
Namely, gathering a collection of graphs that are undirected, weighted, large, and representative of various types of graphs represented several hours of careful work. 
Moreover, devising ways to store these graphs, which were in many cases extremely large, and efficiently load their contents into the programs was an active challenge. 
During the testing phase, we encountered correctness issues that would only appear on specific input graphs, and these issues may not have been corrected if not for the comprehensive benchmarking tools we created.

Ultimately, we created a benchmarking script (mstbench) that would generate random graphs of specific structures on the fly and then execute our various implementations several times on these graphs. 
These graphs were generated using various graph generation algorithms from networkx, and written into a custom binary file format. 
These files (which were in many cases several hundred megabytes) were then saved into tmpfs and fed into the different implementations. 
Since simply reading the file was the source of significant overhead, each implementation would read the file once, and then execute the algorithm a specified number of times. 
This reduced the amount of time spent loading generated graphs into memory. 

## References

* [A Generic and Highly Efficient Parallel Variant of Boruvka's Algorithm](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7092783&tag=1)
* [Engineering Massively Parallel MST Algorithms](https://arxiv.org/pdf/2302.12199.pdf)
* [Wait-Free Union-Find implementation](https://github.com/wjakob/dset/tree/master)
* [15-210 MST Notes](https://www.cs.cmu.edu/afs/cs/academic/class/15210-s15/www/lectures/mst-notes.pdf)
* [Networkx Graph Generators](https://networkx.org/documentation/stable/reference/generators.html)
* [ParlayLib](https://github.com/cmuparlay/parlaylib)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
* [Wikipedia: Boruvka's Algorithm](https://en.wikipedia.org/wiki/Bor%C5%AFvka's_algorithm)


## Goals & Deliverables

### Goals
We are aiming to complete at least 3 of the following 4 goals:
- [x] Implement an optimized sequential version of MST using Baruvka’s algorithm for benchmarking the parallel version against.
- [x] Implement a CPU parallel version of Baruvka’s in parlaylib that can be used for further benchmarking.
- [x] Implement a parallel version of MST in CUDA that scales near-linearly as more GPU threads are used.
- [x] Implement at least 2 meaningfully different versions of a concurrent disjoint-set. Gather significant benchmark data for a parallel implementation of Boruvka’s running using these different implementations.

### Additional/Stretch Goals
- [ ] Implement a parallel version of MST that performs close to, or better than, the existing implementations in Open MP and MPI.
- [ ] Implement an MST-approximation algorithm that performs better than existing MST implementations while getting close to optimal results.

## Current Status

### 2024-05-05
Overall, this project seems to be a success. 
The CUDA implementation managed to get a significant speedup over both the sequential and the ParlayLib implementations for sufficiently large graphs. 
Moreover, a majority of the remaining issues with the speed of the CUDA implementation seem intrinsic to the algorithm or the size of the data being operated on. 

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
~~We will have to expend considerable effort to mitigate this issue, and may need to pivot our project to instead focus on benchmarking different implementations for parallel disjoint-set and how they affect the runtime of our implementations.~~
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
| 4            | Apr 19       | ~~Obtain (any) speedup for finding cheapest edge in CUDA - Jake~~ |
|              | Apr 22       | ~~Implement concurrent disjoint-set data structure in CUDA (ensuring correctness) - Shubham~~ |
| 5            | Apr 25       | ~~Obtain (any) speedup for graph contraction - Jake~~ |
|              | Apr 29       | ~~Have working CUDA code with noticeable speedup. Begin dataset collection - Shubham~~ |
| 6            | May 1        | ~~Fine tune optimizations on CUDA code (remove redundant vertex/edge computation), finalize dataset collection - Jake~~ |
|              | May 5        | ~~Gather all metrics and benchmark data on GHC (stretch: PSC as well) - Shubham~~ |
|              | May 6\*      | Final poster presentation |
